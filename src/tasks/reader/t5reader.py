from numpy import recarray
import torch
import math
from typing import Union
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput

class MaskProbability(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        zero_mask = (p == 0)
        ctx.save_for_backward(zero_mask)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        zero_mask, = ctx.saved_tensors
        grad_output = grad_output.masked_fill(zero_mask, 0.0)
        return grad_output, 

class Reader(torch.nn.Module):
    before_context_pattern = {
        'title': 'Question: {question:s} Title: {title:s} Context: ',
        'no_title': 'Question: {question:s} Context: ',
    }
    no_answer_token = 1 # magic number ,refers to </s>
    return_token_type_ids = False
    def __init__(self, args):
        super().__init__()
        model_name_or_path = args.t5_model_path
        self.topk = args.topk_retrievals
        self.config = T5Config.from_pretrained(model_name_or_path, dropout=args.reader_dropout)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=self.config)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
        # self.with_extractive_loss = (args.with_extractive_loss and args.inference_method != 'select_generative')
        self.with_extractive_loss = args.with_extractive_loss
        self.with_generative_loss = args.with_generative_loss
        self.with_rerank_loss = args.with_rerank_loss
        self.with_pdr_loss = args.with_pdr_loss
        self.with_passage_loss = args.with_passage_loss
        self.extractive_loss_lambda = args.extractive_loss_lambda
        self.extractive_loss_temperature = args.extractive_loss_temperature
        self.inference_method = args.inference_method
        self.support_sentence_length = args.support_sentence_length
        self.support_sentence_topk_accuracies = args.support_sentence_topk_accuracies
        self.rerank_topk_accuracies = args.rerank_topk_accuracies
        self.args = args
        if args.with_extractive_loss:
            self.qa_outputs = torch.nn.Linear(self.model.config.hidden_size, 2) # start, end
            if args.with_passage_loss:
                self.qa_classifier = torch.nn.Linear(self.model.config.hidden_size, 2) # 1 is span answer, 0 is no answer
        else:
            if args.with_rerank_loss:
                # abaltion
                self.qa_classifier = torch.nn.Linear(self.model.config.hidden_size, 2) # 1 is span answer, 0 is no answer
        self.gradient_checkpointing = args.gradient_checkpointing
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def forward(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            labels=None,
            local_have_answer: torch.Tensor = None, # batch_size, topk
            local_start_positions: torch.Tensor = None, # batch_size, topk, max_local_answers
            local_end_positions: torch.Tensor = None, # batch_size, topk, max_local_answers
            local_mask: torch.Tensor = None, # batch_size, topk, max_local_answers
            global_start_positions: torch.Tensor = None, # batch_size, max_global_answers
            global_end_positions: torch.Tensor = None, # batch_size, max_global_answers
            global_mask: torch.Tensor = None, # batch_size, max_global_answers
            have_answers: torch.Tensor = None, # use for rerank
            return_logits = False,
            return_encoder_outputs = False,
            return_decoder_outputs = False,
            record_cross_attention = None,
        ):
        # Convert encoder inputs in embeddings if needed
        record_cross_attention = self.args.record_cross_attention if record_cross_attention is None else record_cross_attention
        batch_size, topk_retrievals, encoder_seq_len = input_ids.shape
        input_ids = input_ids.view(batch_size * topk_retrievals, encoder_seq_len)
        attention_mask = attention_mask.view(batch_size * topk_retrievals, encoder_seq_len)
        input_embeds = self.model.get_input_embeddings()(input_ids)
        encoder_outputs = self.model.encoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        ) # batch_size, topk, seq_len, hidden_size
        encoder_outputs = encoder_outputs.last_hidden_state
        encoder_outputs = encoder_outputs.view(batch_size, topk_retrievals, encoder_seq_len, -1)
        if self.with_pdr_loss and self.with_extractive_loss:
            hidden_size = input_embeds.shape[-1]
            noise = torch.normal(
                mean=torch.zeros(batch_size * topk_retrievals, encoder_seq_len, hidden_size),
                std=torch.ones(batch_size * topk_retrievals, encoder_seq_len, hidden_size) * 1e-3)
            noise = noise.to(input_embeds.device)
            noise.requires_grad_()
            noise_input_embeds = input_embeds + noise
            # print(noise_input_embeds - input_embeds)
            noise_encoder_outputs = self.model.encoder(
                inputs_embeds=noise_input_embeds,
                attention_mask=attention_mask,
            )
            noise_encoder_outputs = noise_encoder_outputs.last_hidden_state
            noise_encoder_outputs = noise_encoder_outputs.view(batch_size, topk_retrievals, encoder_seq_len, -1)
            # print("input embebs:", input_embeds)
            # print("noise input embeds", noise_input_embeds)
            # print("encoder output:", encoder_outputs)
            # print("noise encoder output:", noise_encoder_outputs)
            # print('grad', self.model.shared.weight.grad)
            # input()
        else:
            noise_encoder_outputs = None
        return_dict = {}
        loss = None
        extractive_loss = None
        generative_loss = None
        rerank_loss = None
        pdr_loss = None
        loss = 0
        if self.with_rerank_loss:
            selected_logits, rerank_loss = self.rerank_forward(encoder_outputs, attention_mask, have_answers)
            loss += rerank_loss * self.extractive_loss_lambda
        if self.with_extractive_loss:
            extractive_loss, pdr_loss = self.extractive_forward(
                encoder_outputs, noise_encoder_outputs, attention_mask, local_have_answer,
                local_start_positions, local_end_positions, local_mask,
                global_start_positions, global_end_positions, global_mask)
            loss += extractive_loss * self.extractive_loss_lambda
            if self.with_pdr_loss:
                loss += 4 * pdr_loss * self.extractive_loss_lambda
        if self.with_generative_loss:
            hidden_size = encoder_outputs.shape[-1]
            if self.inference_method == 'select_generative':
                # select important sentences before generate
                encoder_outputs, attention_mask, _, _ = self.select_encoder_outputs(encoder_outputs, attention_mask)
            elif self.inference_method == 'rerank_generative':
                encoder_outputs, attention_mask, _ = self.rerank_encoder_outputs(encoder_outputs, attention_mask)
            elif self.inference_method == "fid_light_generative":
                encoder_outputs, attention_mask = self.fid_light_encoder_outputs(encoder_outputs, attention_mask, self.args.fid_light_k)
            else:
                encoder_outputs = encoder_outputs.view(batch_size, topk_retrievals * encoder_seq_len, hidden_size)
                attention_mask = attention_mask.view(batch_size, topk_retrievals * encoder_seq_len)
            if decoder_input_ids is None:
                decoder_input_ids = self._shift_right(labels)
            # Decode
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                past_key_values=None,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=attention_mask,
                use_cache=not self.gradient_checkpointing,
                output_attentions=record_cross_attention,
                output_hidden_states=None,
                return_dict=None,
            )
            if record_cross_attention:
                scores = decoder_outputs.cross_attentions
                cross_attention_masks = attention_mask[:, None, :] * decoder_attention_mask[:, :, None]
            decoder_outputs = decoder_outputs.last_hidden_state
            # Set device for model parallelism
            if self.model.model_parallel:
                torch.cuda.set_device(self.model.encoder.first_device)
                self.model.lm_head = self.model.lm_head.to(self.model.encoder.first_device)
                decoder_outputs = decoder_outputs.to(self.model.lm_head.weight.device)
            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                decoder_outputs = decoder_outputs * (self.config.d_model**-0.5)
            lm_logits = self.model.lm_head(decoder_outputs)
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                generative_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss += generative_loss
        return_dict = {
            'loss': loss,
            'extractive_loss': extractive_loss,
            'pdr_loss': pdr_loss,
            'generative_loss': generative_loss,
            'rerank_loss': rerank_loss,
        }
        if return_logits:
            return_dict['logits'] = lm_logits
        if return_encoder_outputs:
            return_dict['encoder_last_hidden_state'] = encoder_outputs
        if return_decoder_outputs:
            return_dict['last_hidden_state'] = decoder_outputs
        if record_cross_attention:
            return_dict['cross_attentions'] = scores
            return_dict['cross_attention_masks'] = cross_attention_masks
        return return_dict
    
    def register_cross_attention_hook(self):
        pass
    
    def _extractive_forward(self, encoder_outputs, attention_mask):
        batch_size, topk, seq_len, hidden_size = encoder_outputs.shape
        attention_mask = attention_mask.view(batch_size, topk, seq_len)
        logits = self.qa_outputs(encoder_outputs) # batch_size, topk, seq_len, 2
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) # batch_size, topk, seq_len
        end_logits = end_logits.squeeze(-1) # batch_size, topk, seq_len
        start_logits = start_logits + 1e10 * (attention_mask - 1)
        end_logits = end_logits + 1e10 * (attention_mask - 1)
        selected_logits = None
        if self.with_passage_loss:
            selected_logits = self.qa_classifier(torch.sum(encoder_outputs * attention_mask.unsqueeze(-1), dim=2) / attention_mask.unsqueeze(-1).count_nonzero(dim=2)) # batch_size, topk, 2
        return start_logits, end_logits, selected_logits
    
    def _extractive_probs(self, start_logits, end_logits, selected_logits):
        batch_size, topk, seq_len = start_logits.shape
        start_logits = start_logits.view(batch_size, topk * seq_len)
        end_logits = end_logits.view(batch_size, topk * seq_len)
        start_probs = torch.nn.Softmax(dim=-1)(start_logits)
        end_probs = torch.nn.Softmax(dim=-1)(end_logits)
        start_probs = start_probs.view(batch_size, topk, seq_len)
        end_probs = end_probs.view(batch_size, topk, seq_len)
        selected_probs = None
        if selected_logits is not None:
            selected_probs = torch.nn.Softmax(dim=-1)(selected_logits)
        return start_probs, end_probs, selected_probs
    
    def _extract_start_end_by_prob(self, start_probs, end_probs, selected_probs, attention_mask):
        batch_size, topk_retrievals, encoder_seq_len = start_probs.shape
        start_probs = start_probs.view(batch_size * topk_retrievals, encoder_seq_len, 1)
        end_probs = end_probs.view(batch_size * topk_retrievals, 1, encoder_seq_len)
        if selected_probs is not None:
            selected_probs = selected_probs[:, :, 1].view(batch_size * topk_retrievals, 1, 1)
            span_probs = torch.bmm(start_probs, end_probs) * selected_probs
        else:
            span_probs = torch.bmm(start_probs, end_probs)
        attention_mask = attention_mask.view(batch_size * topk_retrievals, encoder_seq_len)
        attention_mask = attention_mask.to(torch.float32)
        extended_attention_mask = torch.bmm(attention_mask.unsqueeze(2), attention_mask.unsqueeze(1))
        span_probs = span_probs * extended_attention_mask
        sentence_length_mask = torch.triu(torch.ones(encoder_seq_len, encoder_seq_len)) - torch.triu(torch.ones(encoder_seq_len, encoder_seq_len), diagonal=self.support_sentence_length)
        sentence_length_mask = sentence_length_mask.cuda()
        span_probs = span_probs * sentence_length_mask.unsqueeze(0)
        flattern_span_probs = span_probs.contiguous().view(batch_size, -1) # topk * seq_len * seq_len
        topk_probs, topk_indices = flattern_span_probs.topk(k=max(self.support_sentence_topk_accuracies))
        topk_doc_index = torch.floor(topk_indices / (encoder_seq_len * encoder_seq_len)).to(torch.int)
        topk_start_end = topk_indices.fmod(encoder_seq_len * encoder_seq_len)
        topk_start = torch.floor(topk_start_end / encoder_seq_len).to(torch.int)
        topk_end = topk_start_end.fmod(encoder_seq_len)
        return topk_doc_index, topk_start, topk_end, topk_probs

    def extractive_forward(
            self,
            encoder_outputs: torch.Tensor, # batch_size, topk, seq_len, hidden_size
            noise_encoder_outputs: Union[torch.Tensor, None], # batch_size, topk, seq_len, hidden_size
            attention_mask: torch.Tensor, # batch_size, topk, seq_len
            local_have_answer: torch.Tensor, # batch_size, topk
            local_start_positions: torch.Tensor, # batch_size, topk, max_local_answers
            local_end_positions: torch.Tensor, # batch_size, topk, max_local_answers
            local_mask: torch.Tensor, # batch_size, topk, max_local_answers
            global_start_positions: torch.Tensor, # batch_size, max_global_answers
            global_end_positions: torch.Tensor, # batch_size, max_global_answers
            global_mask: torch.Tensor, # batch_size, max_global_answers
        ):
        batch_size, topk, seq_len, hidden_size = encoder_outputs.shape
        start_logits, end_logits, selected_logits = self._extractive_forward(encoder_outputs, attention_mask)
        start_logits = start_logits / self.extractive_loss_temperature
        end_logits = end_logits / self.extractive_loss_temperature
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        # local context use marginal log-likelihood for all correct spans
        start_logits = start_logits.view(batch_size * topk, seq_len) # batch_size * topk, seq_len
        end_logits = end_logits.view(batch_size * topk, seq_len)
        local_start_positions = local_start_positions.view(batch_size * topk, -1) # batch_size * topk, max_local_answers
        local_end_positions = local_end_positions.view(batch_size * topk, -1)
        local_mask = local_mask.view(batch_size * topk, -1)
        local_start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                            for (_start_positions, _span_mask) \
                            in zip(torch.unbind(local_start_positions, dim=-1), torch.unbind(local_mask, dim=-1))]
        local_end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                            for (_end_positions, _span_mask) \
                            in zip(torch.unbind(local_end_positions, dim=-1), torch.unbind(local_mask, dim=-1))]
        assert len(local_start_losses) == len(local_end_losses)
        local_loss_tensor = torch.cat([t.unsqueeze(1) for t in local_start_losses], dim=1) \
                            + torch.cat([t.unsqueeze(1) for t in local_end_losses], dim=1) # batch_size * topk, max_local_answers
        # global context use Hard-EM
        if self.with_passage_loss:
            selected_logits = selected_logits / self.extractive_loss_temperature
            selected_log_probs = torch.unbind(-torch.log(torch.nn.Softmax(dim=-1)(selected_logits)), dim=-1)[1] # batch_size, topk
            selected_log_probs = selected_log_probs.unsqueeze(-1).expand(-1, -1, seq_len) # batch_size, topk, seq_len
            selected_log_probs = selected_log_probs.contiguous()
            selected_log_probs = selected_log_probs.view(batch_size, topk * seq_len) # batch_size, topk * seq_len
            start_logits = start_logits.view(batch_size, topk * seq_len)
            end_logits = end_logits.view(batch_size, topk * seq_len)
            global_start_losses = [((loss_fct(start_logits, _start_positions) \
                                    + selected_log_probs.gather(1, _start_positions.unsqueeze(-1)).squeeze(-1)) \
                                    * _span_mask) \
                                for (_start_positions, _span_mask) \
                                in zip(torch.unbind(global_start_positions, dim=-1), torch.unbind(global_mask, dim=-1))]
            global_end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                                for (_end_positions, _span_mask) \
                                in zip(torch.unbind(global_end_positions, dim=-1), torch.unbind(global_mask, dim=-1))]
        else:
            start_logits = start_logits.view(batch_size, topk * seq_len)
            end_logits = end_logits.view(batch_size, topk * seq_len)
            global_start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                                for (_start_positions, _span_mask) in zip(torch.unbind(global_start_positions, dim=-1), torch.unbind(global_mask, dim=-1))]
            global_end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                                for (_end_positions, _span_mask) \
                                in zip(torch.unbind(global_end_positions, dim=-1), torch.unbind(global_mask, dim=-1))]
        assert len(global_start_losses) == len(global_start_losses)
        global_loss_tensor = torch.cat([t.unsqueeze(1) for t in global_start_losses], dim=1) \
                             + torch.cat([t.unsqueeze(1) for t in global_end_losses], dim=1) # batch_size, max_global_answers
        local_loss_tensor = local_loss_tensor.view(batch_size, topk, -1)
        local_mml_loss = self._take_mml(local_loss_tensor)
        global_hardem_loss = self._take_min(global_loss_tensor)
        # selected loss
        # selected_logits = self.qa_classifier(torch.mean(encoder_outputs * attention_mask.unsqueeze(-1), dim=2)) # batch_size, topk, 2
        if self.with_passage_loss:
            selected_loss = loss_fct(selected_logits.view(batch_size * topk, -1), local_have_answer.view(batch_size * topk))
            selected_loss = selected_loss.view(batch_size, topk)
            selected_loss = torch.mean(selected_loss, dim=-1)
            # print(local_mml_loss.shape, global_hardem_loss.shape, selected_loss.shape)
            extractive_loss = torch.mean(local_mml_loss + global_hardem_loss + selected_loss)
        else:
            extractive_loss = torch.mean(local_mml_loss + global_hardem_loss)
        if self.with_pdr_loss:
            pdr_loss = self._take_pdr_loss(start_logits, end_logits, noise_encoder_outputs, attention_mask)
        else:
            pdr_loss = None
        return extractive_loss, pdr_loss

    def _take_min(self, loss_tensor):
        return torch.min(loss_tensor + 2*torch.max(loss_tensor)*(loss_tensor==0).float(), 1)[0]
    
    def _take_mml(self, loss_tensor):
        # sum version: BUG
        # return -torch.sum(torch.log(torch.sum(torch.exp(-loss_tensor - 1e10 * (loss_tensor==0).float()), -1)), dim=1)
        # mean version
        logp = -loss_tensor - 1e10 * (loss_tensor==0).float() # batch_size, topk, answer_number    
        return -torch.mean(torch.log(torch.sum(torch.exp(logp), -1)), dim=1)
    
    def _take_pdr_loss(self, start_logits, end_logits, noise_encoder_outputs, attention_mask):
        batch_size, topk, seq_len, _ = noise_encoder_outputs.shape
        noise_start_logits, noise_end_logits, _ = self._extractive_forward(noise_encoder_outputs, attention_mask)
        noise_start_logits = noise_start_logits.view(batch_size, topk * seq_len)
        noise_end_logits = noise_end_logits.view(batch_size, topk * seq_len)
        start_logits = start_logits.view(batch_size, topk * seq_len)
        end_logits = end_logits.view(batch_size, topk * seq_len)
        attention_mask = attention_mask.view(batch_size, topk * seq_len)
        fct = torch.nn.Softmax(dim=-1)
        start_probs = MaskProbability.apply(fct(start_logits))
        end_probs = MaskProbability.apply(fct(end_logits))
        noise_start_probs_0 = fct(noise_start_logits)
        noise_start_probs = MaskProbability.apply(noise_start_probs_0)
        noise_end_probs = MaskProbability.apply(fct(noise_end_logits))
        def hellinger_distance(p, q):
            return (p.sqrt() - q.sqrt()).norm(dim=1) / math.sqrt(2)
        pdr_loss = hellinger_distance(start_probs, noise_start_probs) + hellinger_distance(end_probs, noise_end_probs)
        pdr_loss = torch.mean(pdr_loss)
        return pdr_loss
    
    def rerank_forward(self, encoder_outputs, attention_mask, have_answers=None):
        batch_size, topk_retrievals, encoder_seq_len, hidden_size = encoder_outputs.shape
        encoder_outputs = encoder_outputs.view(batch_size * topk_retrievals, encoder_seq_len, -1) # b * t, sl, hidden_size
        selected_logits = self.qa_classifier(torch.sum(encoder_outputs * attention_mask.unsqueeze(-1), dim=1) / attention_mask.unsqueeze(-1).count_nonzero(dim=1)) # b * t, 2
        rerank_loss = None
        if have_answers is not None:
            have_answers = have_answers.view(-1)
            rerank_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(selected_logits, have_answers)
        return selected_logits, rerank_loss
    
    @torch.no_grad()
    def generate(
            self,
            input_ids,
            attention_mask,
            timer=None,
            record_cross_attention=None,
            **kwargs
        ):
        record_cross_attention = self.args.record_cross_attention if record_cross_attention is None else record_cross_attention
        # Convert encoder inputs in embeddings if needed
        if timer is not None and timer('encoder') is not None:
            timer('encoder').start()
        batch_size, topk_retrievals, encoder_seq_len = input_ids.shape
        input_ids = input_ids.view(batch_size * topk_retrievals, encoder_seq_len) 
        attention_mask = attention_mask.view(batch_size * topk_retrievals, encoder_seq_len)
        encoder_outputs_dict = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=record_cross_attention,
        )
        if timer is not None and timer('encoder') is not None:
            timer('encoder').stop()
        encoder_outputs = encoder_outputs_dict.last_hidden_state
        encoder_outputs = encoder_outputs.view(batch_size, topk_retrievals, encoder_seq_len, -1)
        hidden_size = encoder_outputs.shape[-1]
        if self.inference_method == 'generative' or self.inference_method == "fid_light_generative":
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()
            if self.inference_method == "generative":
                encoder_outputs = encoder_outputs.view(batch_size, topk_retrievals * encoder_seq_len, hidden_size)
                attention_mask = attention_mask.view(batch_size, topk_retrievals * encoder_seq_len)
            else:
                encoder_outputs, attention_mask = self.fid_light_encoder_outputs(encoder_outputs, attention_mask, self.args.fid_light_k)
            # Set device for model parallelism
            if self.model.model_parallel:
                torch.cuda.set_device(self.model.decoder.first_device)
                encoder_outputs = encoder_outputs.to(self.model.decoder.first_device)
                decoder_input_ids = decoder_input_ids.to(self.model.decoder.first_device)
                attention_mask = attention_mask.to(self.model.decoder.first_device)
            encoder_outputs_dict.last_hidden_state = encoder_outputs
            decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.int64).cuda()
            output = self.model.generate(
                inputs=decoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs_dict,
                output_attentions=record_cross_attention,
                return_dict_in_generate=True,
                synced_gpus=self.args.synced_gpus,
                **kwargs,
            )
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            output_ids = output.sequences
            return_dict = {
                'sequences': output_ids,
            }
            if record_cross_attention:
                cross_attentions = output.cross_attentions
                decoder_attention_mask = (output_ids[:, 1:] != 0).to(torch.int64)
                cross_attention_masks = decoder_attention_mask[:, :, None] * attention_mask[:, None, :]
                new_cross_attentions = []
                for layer_num in range(len(cross_attentions[0])):
                    layer_cross_attention = torch.cat([c[layer_num] for c in cross_attentions], dim=2) # 4, 12, 7, 9600
                    new_cross_attentions.append(layer_cross_attention)
                    return_dict['cross_attentions'] = new_cross_attentions
                    return_dict['cross_attention_masks'] = cross_attention_masks
                if self.with_extractive_loss:
                    # record select mask
                    encoder_outputs = encoder_outputs.view(batch_size, topk_retrievals, encoder_seq_len, hidden_size)
                    attention_mask = attention_mask.view(batch_size, topk_retrievals, encoder_seq_len)
                    select_masks = self.get_mask_by_encoder_outputs(encoder_outputs, attention_mask)
                    return_dict['select_masks'] = select_masks # batch_size, topk * encoder_seq_len
            return return_dict, None
        elif self.inference_method == 'extractive':
            start_logits, end_logits, selected_logits = self._extractive_forward(encoder_outputs, attention_mask)
            start_probs, end_probs, selected_probs = self._extractive_probs(start_logits, end_logits, selected_logits)
            topk_doc_index, topk_start, topk_end, topk_probs = self._extract_start_end_by_prob(start_probs, end_probs, selected_probs, attention_mask)
            # return start_probs, end_probs, selected_probs
            return topk_doc_index, topk_start, topk_end, topk_probs
        elif self.inference_method == 'select_generative':
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()
            # select important sentences before generate
            select_encoder_outputs_pt, select_encoder_outputs_mask, cross_attention_length, max_cross_attention_length = self.select_encoder_outputs(encoder_outputs, attention_mask)
            decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.int64).cuda()
            encoder_outputs_dict.last_hidden_state = select_encoder_outputs_pt
            output = self.model.generate(
                inputs=decoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=select_encoder_outputs_mask,
                encoder_outputs=encoder_outputs_dict,
                synced_gpus=self.args.synced_gpus,
                **kwargs,
            )
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            return {
                'sequences': output,
            }, (cross_attention_length, max_cross_attention_length)
            # batch_size, k
        elif self.inference_method == 'rerank_generative':
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()
            # select important sentences before generate
            select_encoder_outputs_pt, select_encoder_outputs_mask, cross_attention_length = self.rerank_encoder_outputs(encoder_outputs, attention_mask)
            # print("Genrate:", select_encoder_outputs_pt.shape, select_encoder_outputs_mask.shape)
            decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.int64).cuda()
            encoder_outputs_dict.last_hidden_state = select_encoder_outputs_pt
            output = self.model.generate(
                inputs=decoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=select_encoder_outputs_mask,
                encoder_outputs=encoder_outputs_dict,
                synced_gpus=self.args.synced_gpus,
                **kwargs,
            )
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            return {
                'sequences': output,
            }, (cross_attention_length, cross_attention_length)
            # batch_size, k
        else:
            raise ValueError("No inference method name " + self.inference_method)
        return None
    
    def get_mask_by_start_end(self, doc_index, starts, ends, topk_retrievals, encoder_seq_len):
        mask = torch.zeros(topk_retrievals * encoder_seq_len)
        for doc_i, s, e in zip(doc_index, starts, ends):
            # print(doc_i, s, e)
            mask[doc_i * encoder_seq_len + s: doc_i * encoder_seq_len + e + 1] = 1
        mask = mask.cuda().view(topk_retrievals * encoder_seq_len, 1).to(torch.bool)
        # encoder_outputs = encoder_outputs.view(topk_retrievals * encoder_seq_len, hidden_size)
        # select_encoder_outputs = encoder_outputs.masked_select(mask).view(-1, hidden_size)
        # print(select_encoder_outputs.shape)
        return mask
    
    def get_mask_by_encoder_outputs(self, encoder_outputs, attention_mask):
        # get extraction mask by encoder_outputs
        batch_size, topk_retrievals, encoder_seq_len, hidden_size = encoder_outputs.shape
        start_logits, end_logits, selected_logits = self._extractive_forward(encoder_outputs, attention_mask)
        start_probs, end_probs, selected_probs = self._extractive_probs(start_logits, end_logits, selected_logits)
        topk_doc_index, topk_start, topk_end, topk_probs = self._extract_start_end_by_prob(start_probs, end_probs, selected_probs, attention_mask)
        select_masks = []
        for i in range(batch_size):
            select_mask = self.get_mask_by_start_end(topk_doc_index[i], topk_start[i], topk_end[i], topk_retrievals, encoder_seq_len)
            select_masks.append(select_mask.squeeze(-1))
        select_masks = torch.stack(select_masks, dim=0).to(encoder_outputs.device)
        return select_masks
    
    def select_encoder_outputs(self, encoder_outputs, attention_mask):
        # select important sentences before generate
        batch_size, topk_retrievals, encoder_seq_len, hidden_size = encoder_outputs.shape
        start_logits, end_logits, selected_logits = self._extractive_forward(encoder_outputs, attention_mask)
        start_probs, end_probs, selected_probs = self._extractive_probs(start_logits, end_logits, selected_logits)
        topk_doc_index, topk_start, topk_end, topk_probs = self._extract_start_end_by_prob(start_probs, end_probs, selected_probs, attention_mask)
        select_encoder_outputs = []
        for i in range(batch_size):
            select_mask = self.get_mask_by_start_end(topk_doc_index[i], topk_start[i], topk_end[i], topk_retrievals, encoder_seq_len)
            select_encoder_output = encoder_outputs[i].view(topk_retrievals * encoder_seq_len, hidden_size).masked_select(select_mask).view(-1, hidden_size)
            select_encoder_outputs.append(select_encoder_output)
        select_length = [o.shape[0] for o in select_encoder_outputs]
        select_encoder_outputs_pt = torch.zeros((batch_size, max(select_length), hidden_size), dtype=encoder_outputs.dtype).cuda()
        select_encoder_outputs_mask = torch.zeros((batch_size, max(select_length)), dtype=attention_mask.dtype).cuda()
        cross_attention_length = []
        cross_attention_length.extend(select_length)
        max_cross_attention_length = []
        max_cross_attention_length.extend([max(select_length)] * batch_size)
        for i in range(batch_size):
            current_length = select_length[i]
            select_encoder_outputs_pt[i, :current_length, :] = select_encoder_outputs[i]
            select_encoder_outputs_mask[i, :current_length] = 1
        return select_encoder_outputs_pt, select_encoder_outputs_mask, cross_attention_length, max_cross_attention_length
    
    def rerank_encoder_outputs(self, encoder_outputs, attention_mask):
        # rerank passages before generate
        batch_size, topk_retrievals, encoder_seq_len, hidden_size = encoder_outputs.shape
        selected_logits, _ = self.rerank_forward(encoder_outputs, attention_mask)
        selected_logits = selected_logits.view(batch_size, topk_retrievals, -1)
        selected_logits = selected_logits[:, :, 1]
        topk_logits, topk_indices = torch.topk(selected_logits, k=max(self.rerank_topk_accuracies), dim=1)
        gather_index = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, encoder_seq_len, hidden_size)
        rerank_encoder_outputs = torch.gather(encoder_outputs, dim=1, index=gather_index)
        rerank_encoder_outputs = rerank_encoder_outputs.view(-1, max(self.rerank_topk_accuracies) * encoder_seq_len, hidden_size)
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, encoder_seq_len)
        attention_mask = attention_mask.view(batch_size, topk_retrievals, encoder_seq_len)
        rerank_attention_mask = torch.gather(attention_mask, dim=1, index=gather_index)
        rerank_attention_mask = rerank_attention_mask.view(-1, max(self.rerank_topk_accuracies) * encoder_seq_len)
        cross_attention_length = [max(self.rerank_topk_accuracies) * encoder_seq_len] * batch_size
        return rerank_encoder_outputs, rerank_attention_mask, cross_attention_length

    def fid_light_encoder_outputs(self, encoder_outputs, attention_mask, fid_light_k):
        # FiD-Light: extract first k tokens in each input
        batch_size, topk_retrievals, encoder_seq_len, hidden_size = encoder_outputs.shape
        attention_mask = attention_mask.view(batch_size, topk_retrievals, encoder_seq_len)
        encoder_outputs = encoder_outputs[:, :, :fid_light_k, :].contiguous()
        attention_mask = attention_mask[:, :, :fid_light_k].contiguous()
        encoder_outputs = encoder_outputs.view(batch_size, topk_retrievals * fid_light_k, hidden_size)
        attention_mask = attention_mask.view(batch_size, topk_retrievals * fid_light_k)
        return encoder_outputs, attention_mask
        
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.model.config.decoder_start_token_id
        pad_token_id = self.model.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids