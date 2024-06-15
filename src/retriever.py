import logging
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import transformers
from transformers import BertModel, BertTokenizer
from transformers import (DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
                          DPRContextEncoder, DPRContextEncoderTokenizer)

logger = logging.getLogger("Retriever")

class Retriever(object):
    def __init__(self):
        pass
    
    def embeded_text(self, input_ids, attention_mask, token_type_ids):
        pass
    
    def encode_question(self, input_ids, attention_mask, token_type_ids):
        pass


class DPRRetriever(Retriever):
    '''
        Dual Encoder Retriever
    '''
    passage_format = "{title} [SEP] {text}"
    def __init__(self, args):
        self.query_encoder = None
        self.context_encoder = None
        self.no_extract_cls = args.no_extract_cls
        if args.query_encoder_path is not None:
            self.query_encoder = DPRQuestionEncoder.from_pretrained(args.query_encoder_path)
            self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.query_encoder_path)
        if args.context_encoder_path is not None:
            self.context_encoder = DPRContextEncoder.from_pretrained(args.context_encoder_path)
            self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.context_encoder_path)
        world_size = args.world_size
        rank = args.rank
        if world_size > 1:
            # Data parallel
            logger.warning(f"Running DDP on rank {rank} / {world_size}")
            dist.barrier()
            if self.query_encoder is not None:
                self.query_encoder.to(rank)
                self.query_encoder = DDP(self.query_encoder, device_ids=[rank], output_device=rank)
            if self.context_encoder is not None:
                self.context_encoder.to(rank)
                self.context_encoder = DDP(self.context_encoder, device_ids=[rank], output_device=rank)
            dist.barrier()
        else:
            if self.query_encoder is not None:
                self.query_encoder.cuda()
            if self.context_encoder is not None:
                self.context_encoder.cuda()
    
    def embeded_text(self, input_ids, attention_mask, token_type_ids):
        assert self.context_encoder is not None
        outputs = self.context_encoder(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        logits = outputs.pooler_output
        return logits
    
    def encode_question(self, input_ids, attention_mask, token_type_ids):
        assert self.query_encoder is not None
        outputs = self.query_encoder(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        logits = outputs.pooler_output
        return logits

class FiDKDEncoderConfig(transformers.BertConfig):

    def __init__(self,
                 indexing_dimension=768,
                 apply_question_mask=False,
                 apply_passage_mask=False,
                 extract_cls=False,
                 passage_maxlength=200,
                 question_maxlength=40,
                 projection=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls=extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection

class FiDKDEncoder(transformers.PreTrainedModel):
    config_class = FiDKDEncoderConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)
        assert config.projection or config.indexing_dimension == 768, \
            'If no projection then indexing dimension must be equal to 768'
        self.config = config
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        if self.config.projection:
            self.proj = torch.nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )
            self.norm = torch.nn.LayerNorm(self.config.indexing_dimension)
        self.loss_fct = torch.nn.KLDivLoss()
    
    def _init_weights(self, module):
        """Initialize the weights"""
        self.model._init_weights(module)
    
    def forward(self,
                question_ids,
                question_mask,
                passage_ids,
                passage_mask,
                gold_score=None):
        question_output = self.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_text(
            text_ids=passage_ids,
            text_mask=passage_mask,
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )

        score = torch.einsum(
            'bd,bid->bi',
            question_output,
            passage_output.view(bsz, n_passages, -1)
        )
        score = score / np.sqrt(question_output.size(-1))
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return question_output, passage_output, score, loss

    def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = self.model(
            input_ids=text_ids,
            attention_mask=text_mask if apply_mask else None
        )
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)


class FiDKDRetriever(Retriever):
    '''
    Retriever of FiDKD
    '''
    passage_format = "title: {title} context: {text}"
    
    def __init__(self, args):
        self.query_encoder = None
        self.context_encoder = None
        assert args.context_encoder_path is not None
        self.config = FiDKDEncoderConfig.from_pretrained(args.context_encoder_path)
        self.context_encoder = FiDKDEncoder.from_pretrained(args.context_encoder_path)
        self.context_tokenizer = BertTokenizer.from_pretrained(args.context_encoder_path)
        self.context_encoder.cuda()
        # query encoder is same as context encoder in FiD-KD
        self.query_encoder = self.context_encoder
        self.query_tokenizer = self.context_tokenizer
    
    def embeded_text(self, input_ids, attention_mask, token_type_ids):
        assert self.context_encoder is not None
        logits = self.context_encoder.embed_text(
            text_ids=input_ids,
            text_mask=attention_mask.bool(),
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )
        return logits
    
    def encode_question(self, input_ids, attention_mask, token_type_ids):
        assert self.query_encoder is not None
        logits = self.query_encoder.embed_text(
            text_ids=input_ids,
            text_mask=attention_mask.bool(),
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        return logits
    
    