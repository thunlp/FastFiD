import warnings
import torch
import logging
from typing import Optional, Tuple
from flash_attn import __version__ as flash_attn_version

from models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaAttention, rotate_half
from models.llama.configuration_llama import LlamaConfig
from models.llama.tokenization_llama_fast import LlamaTokenizerFast
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
)

logger = logging.getLogger("llama-reader")

def apply_rotary_pos_emb(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len
    # print(f"rotary: rank: {torch.distributed.get_rank()} kv_seq_len: {kv_seq_len} max_position_ids: {position_ids.max().item()}")
    cos_sin = self.rotary_emb(v, seq_len=position_ids.max().item() + 1)
    q, k = apply_rotary_pos_emb(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        assert (
            flash_attn_version >= "2.1.0"
        ), "past_key_value support requires flash-attn >= 2.1.0"
        # reuse k, v
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None
    dropout_p = self.config.attention_dropout if self.training else 0.0

    if attention_mask is None:
        output = flash_attn_func(q, k, v, dropout_p, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            dropout_p,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean key_padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    # rank = torch.distributed.get_rank()
    # if rank == 0:
    #     print(attention_mask, attention_mask.shape, past_key_values_length)
    # if past_key_values_length > 0 and attention_mask is not None:
    #     attention_mask = torch.cat(
    #         (
    #             torch.full(
    #                 (input_shape[0], past_key_values_length),
    #                 True,
    #                 dtype=attention_mask.dtype,
    #                 device=attention_mask.device,
    #             ),
    #             attention_mask,
    #         ),
    #         dim=-1,
    #     )
    #     if rank == 0:
    #         print("generate:", attention_mask, attention_mask.shape)

    if attention_mask is not None and torch.all(attention_mask):
        # if rank == 0:
        #     print("attention mask to None")
        return None  # This uses the faster call when training with full samples

    return attention_mask


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    LlamaAttention.forward = forward

class Reader(torch.nn.Module):
    before_context_pattern = None
    no_answer_token = None # magit number ,refers to </s>
    return_token_type_ids = None
    def __init__(self, args):
        super().__init__()
        replace_llama_attn_with_flash_attn()
        model_name_or_path = args.t5_model_path
        self.topk = args.topk_retrievals
        self.config = LlamaConfig.from_pretrained(model_name_or_path, attention_dropout=args.reader_dropout, hidden_dropout=args.reader_dropout)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, config=self.config)
        # self.model = LlamaForCausalLM(self.config)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_name_or_path)
        self.args = args
        self.gradient_checkpointing = args.gradient_checkpointing
        self.with_extractive_loss = args.with_extractive_loss
        self.extractive_loss_temperature = args.extractive_loss_temperature
        self.extractive_loss_lambda = args.extractive_loss_lambda
        self.inference_method = args.inference_method
        self.support_sentence_length = args.support_sentence_length
        self.support_sentence_topk_accuracies = args.support_sentence_topk_accuracies
        if args.with_extractive_loss:
            self.qa_dropout = torch.nn.Dropout(args.reader_dropout)
            self.qa_outputs = torch.nn.Linear(self.model.config.hidden_size, 2) # start, end
        if self.gradient_checkpointing:
            if self.inference_method != "select_generative":
                self.model.gradient_checkpointing_enable()
            else:
                logger.warning("Gradient checkpointing is not supported for select_generative inference method with llama reader")
        
    def forward(
            self,
            input_ids,
            attention_mask,
            labels = None,
            only_labels = None,
            only_labels_input = None,
            labels_attention_mask = None,
            prompt_mask = None,
            global_start_positions=None,
            global_end_positions=None,
            global_mask=None,
            context_ids=None,
            context_attention_mask=None,
            return_logits = False,
            return_encoder_outputs = False,
            record_cross_attention = None,
        ):
        # Convert encoder inputs in embeddings if needed
        return_dict = {}
        return_dict["loss"] = 0
        if self.inference_method != "select_generative":
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states = return_encoder_outputs or self.with_extractive_loss,
                return_dict=True
            )
            generative_loss = output.loss
            return_dict["loss"] += generative_loss
            return_dict["generative_loss"]= generative_loss
        else:
            input_ids = context_ids
            attention_mask = context_attention_mask
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                output_hidden_states = return_encoder_outputs or self.with_extractive_loss,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
        if self.with_extractive_loss:
            last_hidden_state = output.hidden_states[-1]
            if labels is not None and self.inference_method != "select_generative":
                ext_attention_mask = attention_mask * (labels == -100)
            else:
                ext_attention_mask = attention_mask
            extractive_loss, start_logits, end_logits = self.extractive_forward(
                last_hidden_state,
                ext_attention_mask,
                global_start_positions,
                global_end_positions,
                global_mask,
            )
            return_dict["loss"] += (extractive_loss * self.extractive_loss_lambda)
            return_dict['extractive_loss'] = extractive_loss
        if self.inference_method == "select_generative":
            ori_length = attention_mask.sum(dim=-1).tolist()
            batch_size, label_len = only_labels_input.shape
            new_past_key_values, new_attention_mask, new_lengths, max_new_length = self.select_kv_cache(start_logits, end_logits, past_key_values, context_attention_mask, prompt_mask)
            position_ids = []
            for i in range(batch_size):
                position_ids.append(list(range(ori_length[i], ori_length[i] + label_len)))
            position_ids = torch.tensor(position_ids, dtype=torch.int64, device=only_labels_input.device)
            labels_attention_mask = torch.cat((new_attention_mask, labels_attention_mask), dim=-1)
            # if self.args.rank == 0:
            #     print("position_ids:", position_ids)
            #     print("labels_attention_mask:", labels_attention_mask.shape)
            output = self.model(
                input_ids=only_labels_input,
                attention_mask=labels_attention_mask,
                position_ids=position_ids,
                past_key_values=new_past_key_values,
                labels=only_labels,
                output_hidden_states = return_encoder_outputs,
                return_dict=True,
            )
            generative_loss = output.loss
            return_dict["loss"] += generative_loss
            return_dict["generative_loss"]= generative_loss
            
        if return_logits:
            return_dict['logits'] = output.logits # batch_size x seq_len x vocab_size
        if return_encoder_outputs:
            return_dict['encoder_last_hidden_state'] = output.hidden_states[-1]
        return return_dict
    
    def extractive_forward(
        self,
        encoder_outputs: torch.Tensor, # batch_size, seq_len, hidden_size
        attention_mask: torch.Tensor, # batch_size, seq_len
        global_start_positions: torch.Tensor, # batch_size, max_global_answers
        global_end_positions: torch.Tensor, # batch_size, max_global_answers
        global_mask: torch.Tensor, # batch_size, max_global_answers
    ):
        encoder_outputs = self.qa_dropout(encoder_outputs)
        logits = self.qa_outputs(encoder_outputs) # batch_size, seq_len, 2
        start_logits, end_logits = torch.unbind(logits, dim=-1)
        start_logits = start_logits / self.extractive_loss_temperature
        end_logits = end_logits / self.extractive_loss_temperature
        start_logits = start_logits.squeeze(-1) + (attention_mask - 1) * 1e10 # batch_size, seq_len
        end_logits = end_logits.squeeze(-1) + (attention_mask - 1) * 1e10
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        global_start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                            for (_start_positions, _span_mask) in zip(torch.unbind(global_start_positions, dim=-1), torch.unbind(global_mask, dim=-1))]
        global_end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                            for (_end_positions, _span_mask) \
                            in zip(torch.unbind(global_end_positions, dim=-1), torch.unbind(global_mask, dim=-1))]
        assert len(global_start_losses) == len(global_end_losses)
        global_loss_tensor = torch.cat([t.unsqueeze(1) for t in global_start_losses], dim=1) \
                            + torch.cat([t.unsqueeze(1) for t in global_end_losses], dim=1) # bs, max_global_answers
        global_loss = global_loss_tensor.sum(dim=-1) / global_mask.sum(dim=-1)
        # if not self.training:
        #     rank = torch.distributed.get_rank()
        #     if rank == 0:
        #         print(f"rank: {rank} global_loss: {global_loss} global_mask: {global_mask} global_start: {global_start_positions}")
        #         print(f"global start loss: {global_start_losses}")
        #         for (_start_positions, _span_mask) in zip(torch.unbind(global_start_positions, dim=-1), torch.unbind(global_mask, dim=-1)):
        #             print(f"start positions: {_start_positions} span mask: {_span_mask} start_logits: {start_logits.shape}")
        #     torch.distributed.barrier()
        #     quit()
        return global_loss.mean(), start_logits, end_logits
        

    @torch.no_grad()
    def generate(
            self,
            input_ids,
            attention_mask,
            prompt_mask=None,
            timer=None,
            record_cross_attention=None,
            **kwargs
        ):
        batch_size, seq_len = input_ids.shape
        # rank = torch.distributed.get_rank()
        # print(f"rank: {rank} seq_len: {seq_len}")
        if self.gradient_checkpointing and self.inference_method != "select_generative":
            self.model.gradient_checkpointing_disable()
        if self.inference_method == "generative":
            output = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                synced_gpus=self.args.synced_gpus,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
            output_ids = output.sequences
            return_dict = {
                'sequences': output_ids,
            }
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            return return_dict, (attention_mask.sum(dim=-1).tolist(), [seq_len] * batch_size)
        elif self.inference_method == "extractive":
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_state = output.hidden_states[-1]
            logits = self.qa_outputs(last_hidden_state)
            start_logits, end_logits = torch.unbind(logits, dim=-1)
            start_logits = start_logits.squeeze(-1) + (attention_mask - 1) * 1e10 # bs, seq_len
            end_logits = end_logits.squeeze(-1) + (attention_mask - 1) * 1e10
            topk_doc_index, topk_start, topk_end, topk_probs = self.get_topk_spans(start_logits, end_logits)
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            return topk_doc_index, topk_start, topk_end, topk_probs
        elif self.inference_method == "select_generative":
            only_labels_input = input_ids[:, -1:]
            labels_attention_mask = attention_mask[:, -1:]
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )
            last_hidden_state = output.hidden_states[-1]
            past_key_values = output.past_key_values
            start_logits, end_logits = self.qa_outputs(last_hidden_state).unbind(dim=-1)
            start_logits = start_logits.squeeze(-1) + (attention_mask - 1) * 1e10
            end_logits = end_logits.squeeze(-1) + (attention_mask - 1) * 1e10
            ori_length = attention_mask.sum(dim=-1).tolist()
            batch_size, label_len = only_labels_input.shape
            # select kv cache
            new_past_key_values, new_attention_mask, new_lengths, max_new_length = self.select_kv_cache(start_logits, end_logits, past_key_values, attention_mask, prompt_mask)
            num_beams = kwargs.get('num_beams', 1)
            exp_past_key_values = ()
            for layer_idx, (k, v) in enumerate(new_past_key_values):
                exp_past_key_values += ((torch.repeat_interleave(k, num_beams, dim=0), torch.repeat_interleave(v, num_beams, dim=0)), )
            position_ids = []
            for i in range(batch_size):
                position_ids.append(list(range(ori_length[i], ori_length[i] + label_len)))
            position_ids = torch.tensor(position_ids, dtype=torch.int64, device=only_labels_input.device)
            labels_attention_mask = torch.cat((new_attention_mask, labels_attention_mask), dim=-1)
            output = self.model.generate(
                inputs=only_labels_input,
                attention_mask=labels_attention_mask,
                position_ids=position_ids,
                past_key_values=exp_past_key_values,
                return_dict_in_generate=True,
                synced_gpus=self.args.synced_gpus,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
            output_ids = output.sequences
            return_dict = {
                'sequences': output_ids,
            }
            return return_dict, (new_lengths, [max_new_length] * batch_size)
        else:
            raise ValueError(f"Invalid inference method: {self.inference_method}")
    
    def get_topk_spans(self, start_logits, end_logits):
        batch_size, seq_len = start_logits.shape
        start_probs = torch.nn.functional.softmax(start_logits, dim=-1) # bs, seq_len
        end_probs = torch.nn.functional.softmax(end_logits, dim=-1)
        start_probs = start_probs.unsqueeze(-1) # bs, seq_len, 1
        end_probs = end_probs.unsqueeze(-2) # bs, 1, seq_len
        span_probs = torch.bmm(start_probs, end_probs) # bs, seq_len, seq_len
        sentence_length_mask = torch.triu(torch.ones(seq_len, seq_len)) - torch.triu(torch.ones(seq_len, seq_len), diagonal=self.support_sentence_length) # seq_len, seq_len
        sentence_length_mask = sentence_length_mask.cuda()
        span_probs = span_probs * sentence_length_mask.unsqueeze(0)
        flattern_span_probs = span_probs.contiguous().view(batch_size, -1) # bs, seq_len * seq_len
        topk_probs, topk_indices = flattern_span_probs.topk(k=max(self.support_sentence_topk_accuracies))
        topk_start = topk_indices.floor_divide(seq_len).to(torch.int64) # bs, topk
        topk_end = topk_indices.fmod(seq_len).to(torch.int64)
        topk_doc_index = torch.zeros_like(topk_start).to(torch.int64) # for extraction, no meaning
        return topk_doc_index, topk_start, topk_end, topk_probs

    def select_kv_cache(self, start_logits, end_logits, past_key_values, attention_mask, prompt_mask):
        # rank = torch.distributed.get_rank()
        topk_doc_index, topk_start, topk_end, topk_probs = self.get_topk_spans(start_logits, end_logits)
        batch_size, num_heads, seq_len, head_size = past_key_values[0][0].shape
        new_length = []
        new_kv_cache = []
        for i in range(batch_size):
            mask = torch.zeros((seq_len), dtype=torch.bool, device=start_logits.device)
            for s, e in zip(topk_start[i], topk_end[i]):
                mask[s:e + 1] = True
            mask = mask * attention_mask[i] + prompt_mask[i]
            mask = mask.view(1, seq_len, 1).to(torch.bool)
            new_ks = []
            new_vs = []
            for k, v in past_key_values:
                new_k = k[i].masked_select(mask).view(num_heads, -1, head_size)
                new_v = v[i].masked_select(mask).view(num_heads, -1, head_size)
                new_ks.append(new_k)
                new_vs.append(new_v)
            new_kv_cache.append((new_ks, new_vs))
            new_length.append(mask.sum().item())
        # if not self.training:
        #     print(f"rank: {rank} new_length: {new_length} k shape: {new_kv_cache[0][0][0].shape}")
        max_length = max(new_length)
        new_past_key_values = ()
        for i in range(len(past_key_values)):
            new_past_key_values += (([], []),)
        attention_mask = []
        for i in range(batch_size):
            mask = [0] * (max_length - new_length[i]) + [1] * new_length[i]
            attention_mask.append(mask)
            ks, vs = new_kv_cache[i]
            for layer_idx, (k, v) in enumerate(zip(ks, vs)):
                k = torch.cat([torch.zeros((num_heads, max_length - new_length[i], head_size), dtype=k.dtype, device=k.device), k], dim=1)
                v = torch.cat([torch.zeros((num_heads, max_length - new_length[i], head_size), dtype=v.dtype, device=v.device), v], dim=1)
                new_past_key_values[layer_idx][0].append(k)
                new_past_key_values[layer_idx][1].append(v)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64, device=start_logits.device)
        new_past_key_values = tuple((torch.stack(ks, dim=0), torch.stack(vs, dim=0)) for ks, vs in new_past_key_values)
        # if rank == 0:
        #     print("new_kv:", len(new_past_key_values), len(new_past_key_values[0]), new_past_key_values[0][0].shape)
        #     print("new attention mask:", attention_mask, attention_mask.shape)
        return new_past_key_values, attention_mask, new_length, max_length
        