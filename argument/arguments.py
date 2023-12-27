import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class DataArguments:
    """arguments for dataset class"""
    train_query_path: str = field(default='/home/ljb/reorg_lcr/dataset/lecardv2/train_queries.pkl', metadata={"help": "The train query path."})
    corpus_path: str = field(default='/home/ljb/reorg_lcr/dataset/lecardv2/candidates.pkl', metadata={"help": "The corpus path."})
    label_path: str = field(default='/home/ljb/reorg_lcr/dataset/lecardv2/relevance.pkl', metadata={"help": "The label path."})
    hard_negative_path: str = field(default='/home/ljb/lcr/models/bm25/all_bm25_top1000.pkl', metadata={"help": "The hard negative path."})
    hard_negative_num: int = field(default=8, metadata={"help": "The number of hard negative."})
    query_max_len: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization for query."
                  "Sequences longer will be truncated, sequences shorter will be padded."
                }
    )
    doc_max_len: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization for doc."
                  "Sequences longer will be truncated, sequences shorter will be padded."
                }
    )
    def __post_init__(self):
        if not os.path.exists(self.train_query_path):
            raise ValueError(f"train_query_path {self.train_query_path} does not exist.")
        if not os.path.exists(self.corpus_path):
            raise ValueError(f"corpus_path {self.corpus_path} does not exist.")
        if not os.path.exists(self.label_path):
            raise ValueError(f"label_path {self.label_path} does not exist.")
        if not os.path.exists(self.hard_negative_path):
            raise ValueError(f"hard_negative_path {self.hard_negative_path} does not exist.")
        
        

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/home/ljb/models/bert-base-chinese",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    """
    继承了TrainingArguments的全部参数，添加了一些新的参数
    """
    temperature: float = field(default=1.0)
    sentence_pooling_method: str = field(default="cls", metadata={"help": "The pooling method, cls or mean."})
    normalized: bool = field(default=True)