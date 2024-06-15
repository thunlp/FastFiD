import torch
from transformers import ElectraTokenizerFast, ElectraModel
from transformers.modeling_outputs import Seq2SeqLMOutput


class Reader(torch.nn.Module):
    before_context_pattern = {
        'title': '{title:s} [SEP] ',
        'no_title': '',
    }
    no_answer_token = 101 # magit number ,refers to [CLS]
    return_token_type_ids = True
    def __init__(self, args):
        super().__init__()
        model_name_or_path = args.t5_model_path
        self.topk = args.topk_retrievals
        self.t5 = ElectraModel.from_pretrained(model_name_or_path)
        self.t5_tokenizer = ElectraTokenizerFast.from_pretrained(model_name_or_path)
        self.with_extractive_loss = args.with_extractive_loss
        self.with_generative_loss = args.with_generative_loss
        if self.with_generative_loss:
            raise ValueError("Electra can not have generative loss")
        if args.with_extractive_loss:
            self.qa_outputs = torch.nn.Linear(self.t5.config.hidden_size, 2) # start, end
            self.qa_classifier = torch.nn.Linear(self.t5.config.hidden_size, 2) # 1 is span answer, 0 is no answer
    
    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
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
        ):
        # Convert encoder inputs in embeddings if needed
        batch_size, topk_retrievals, encoder_seq_len = input_ids.shape
        input_ids = input_ids.view(batch_size * topk_retrievals, encoder_seq_len) 
        attention_mask = attention_mask.view(batch_size * topk_retrievals, encoder_seq_len)
        token_type_ids = token_type_ids.view(batch_size * topk_retrievals, encoder_seq_len)
        encoder_outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ) # batch_size, topk, seq_len, hidden_size
        encoder_outputs = encoder_outputs.last_hidden_state
        encoder_outputs = encoder_outputs.view(batch_size, topk_retrievals, encoder_seq_len, -1)
        return_dict = {}
        loss = None
        extractive_loss = None
        generative_loss = None
        if self.with_extractive_loss:
            extractive_loss = self.extractive_forward(
                encoder_outputs, attention_mask, local_have_answer,
                local_start_positions, local_end_positions, local_mask,
                global_start_positions, global_end_positions, global_mask)
            if not self.with_generative_loss:
                loss = extractive_loss
        return_dict = {
            'loss': loss,
            'extractive_loss': extractive_loss,
            'generative_loss': generative_loss,
        }
        return return_dict
    
    def _extractive_forward(self, encoder_outputs, attention_mask):
        batch_size, topk, seq_len, hidden_size = encoder_outputs.shape
        attention_mask = attention_mask.view(batch_size, topk, seq_len)
        logits = self.qa_outputs(encoder_outputs) # batch_size, topk, seq_len, 2
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) # batch_size, topk, seq_len
        end_logits = end_logits.squeeze(-1) # batch_size, topk, seq_len
        start_logits = start_logits + 1e10 * (attention_mask - 1)
        end_logits = end_logits + 1e10 * (attention_mask - 1)
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
        selected_probs = torch.nn.Softmax(dim=-1)(selected_logits)
        return start_probs, end_probs, selected_probs

    def extractive_forward(
            self,
            encoder_outputs: torch.Tensor, # batch_size, topk, seq_len, hidden_size
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
        assert len(global_start_losses) == len(global_start_losses)
        global_loss_tensor = torch.cat([t.unsqueeze(1) for t in global_start_losses], dim=1) \
                             + torch.cat([t.unsqueeze(1) for t in global_end_losses], dim=1) # batch_size, max_global_answers
        local_loss_tensor = local_loss_tensor.view(batch_size, topk, -1)
        local_mml_loss = self._take_mml(local_loss_tensor)
        global_hardem_loss = self._take_min(global_loss_tensor)
        # selected loss
        # selected_logits = self.qa_classifier(torch.mean(encoder_outputs * attention_mask.unsqueeze(-1), dim=2)) # batch_size, topk, 2
        selected_loss = loss_fct(selected_logits.view(batch_size * topk, -1), local_have_answer.view(batch_size * topk))
        selected_loss = selected_loss.view(batch_size, topk)
        selected_loss = torch.mean(selected_loss, dim=-1)
        # print(local_mml_loss.shape, global_hardem_loss.shape, selected_loss.shape)
        extractive_loss = torch.mean(local_mml_loss + global_hardem_loss + selected_loss)
        return extractive_loss

    def _take_min(self, loss_tensor):
        return torch.min(loss_tensor + 2*torch.max(loss_tensor)*(loss_tensor==0).float(), 1)[0]
    
    def _take_mml(self, loss_tensor):
        # sum version: BUG
        # return -torch.sum(torch.log(torch.sum(torch.exp(-loss_tensor - 1e10 * (loss_tensor==0).float()), -1)), dim=1)
        # mean version
        return -torch.mean(torch.log(torch.sum(torch.exp(-loss_tensor - 1e10 * (loss_tensor==0).float()), -1)), dim=1)
    
    @torch.no_grad()
    def generate(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            **kwargs
        ):
        # Convert encoder inputs in embeddings if needed
        batch_size, topk_retrievals, encoder_seq_len = input_ids.shape
        input_ids = input_ids.view(batch_size * topk_retrievals, encoder_seq_len) 
        attention_mask = attention_mask.view(batch_size * topk_retrievals, encoder_seq_len)
        token_type_ids = token_type_ids.view(batch_size * topk_retrievals, encoder_seq_len)
        encoder_outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = encoder_outputs.last_hidden_state
        encoder_outputs = encoder_outputs.view(batch_size, topk_retrievals, encoder_seq_len, -1)
        hidden_size = encoder_outputs.shape[-1]
        if self.with_extractive_loss and not self.with_generative_loss:
            start_logits, end_logits, selected_logits = self._extractive_forward(encoder_outputs, attention_mask)
            start_probs, end_probs, selected_probs = self._extractive_probs(start_logits, end_logits, selected_logits)
            return start_probs, end_probs, selected_probs
        if self.with_extractive_loss and self.with_generative_loss:
            raise ValueError("Have not implemented combine")
        if not self.with_extractive_loss and not self.with_generative_loss:
            raise ValueError("You should at least use one mode.")
        return None

