import logging
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None # (batch_size, hidden_size)
    pos_reps: Optional[Tensor] = None # (batch_size, hidden_size)
    negs_reps: Optional[Tensor] = None # (batch_size, hard_negative_num, hidden_size)
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoder(nn.Module):
    def __init__(
            self,
            model_name_or_path: str = None,
            normalized: bool = False,
            sentence_pooling_method: str = "cls",
            temperature: float = 1.0,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.normalized = normalized

        if not normalized:
            self.temperature = 1.0
            logger.info("reset temperature to 1.0 due to using inner product to compute similarity")

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)
        
    def encode(self, features):
        if features is None:
            return None
        out = self.model(**features, return_dict=True)
        if self.sentence_pooling_method == "cls":
            rep = out.last_hidden_state[:, 0, :]
        elif self.sentence_pooling_method == "mean":
            rep = torch.mean(out.last_hidden_state, dim=1)
        if self.normalized:
            rep = torch.nn.functional.normalize(rep, dim=-1)
        return rep
    
    def forward(self, query, pos, negs) -> EncoderOutput:
        """
        params:
            Tuple[BatchEncoding, BatchEncoding, List[BatchEncoding]]
        return:
            EncoderOutput
        """
        q_reps = self.encode(query)  # (batch_size, hidden_size)
        pos_reps = self.encode(pos)  # (batch_size, hidden_size)
        negs_reps = []
        for neg in negs:
            negs_reps.append(self.encode(neg))
        negs_reps = torch.stack(negs_reps, dim=0)  # (batch_size, hard_negative_num, hidden_size)
        
        scores = self.compute_similarity(q_reps, pos_reps, negs_reps)  # (batch_size, hard_negative_num + 1)
        scores = scores / self.temperature  
        if self.training:
            labels = torch.zeros_like(scores, dtype=torch.float)  # (batch_size, hard_negative_num + 1)
            labels[:, 0] = 1.0  # 第一列为正样本

            loss = nn.CrossEntropyLoss()(scores, labels)
            return EncoderOutput(loss=loss, scores=scores)
        else:
            return EncoderOutput(scores=scores, loss=None)
        
    def compute_similarity(self, q_reps, pos_reps, negs_reps):
        """
        计算 query 和 pos/negs 的相似度
        params:
            q_reps: Tensor, (batch_size, hidden_size)
            pos_reps: Tensor, (batch_size, hidden_size)
            negs_reps: Tensor, (batch_size, hard_negative_num, hidden_size)
        return:
            Tensor, (batch_size, hard_negative_num + 1), 第一列为query和pos的相似度，后面为query和negs的相似度
        """
        pos_scores = torch.sum(q_reps * pos_reps, dim=-1)  # (batch_size, )

        # 计算每个 query 和对应 negs 的相似度
        # query: (batch_size, hidden_size), negs: (batch_size, hard_negative_num, hidden_size)
        q_reps = q_reps.unsqueeze(1)  # (batch_size, 1, hidden_size)
        negs_scores = torch.bmm(q_reps, negs_reps.transpose(1, 2))  # (batch_size, 1, hidden_size) * (batch_size, hidden_size, hard_negative_num) ->
                                                                    # -> (batch_size, 1, hard_negative_num)
        negs_scores = negs_scores.squeeze(1)  # (batch_size, hard_negative_num)

        scores = torch.cat([pos_scores.unsqueeze(1), negs_scores], dim=1)  # (batch_size, hard_negative_num + 1)
        return scores

