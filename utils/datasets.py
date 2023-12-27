"""Dataset类"""
import random
import json, pickle
from typing import List, Tuple

from dataclasses import dataclass
from argument.arguments import DataArguments
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer):

        self.query = None
        self.corpus = None
        self.label = None
        self.hard_negative = None
        self.load_data(args)
        self.tokenizer = tokenizer
        self.args = args
        self.pos_pairs = []  # (query, pos)
        self.get_pos_pairs()

    def __len__(self):
        return len(self.pos_pairs)
    
    def __getitem__(self, idx) -> Tuple[str, str, List[str]]:
        """
        Return:
            query: str
            pos: str
            negs: List[str]
        """
        q_id, pos_id = self.pos_pairs[idx]
        query = self.query[q_id]['fact']
        pos = self.corpus[pos_id]['fact']
        negs = []
        for i, doc_id in enumerate(self.hard_negative[q_id]):
            if i >= self.args.hard_negative_num:
                break
            negs.append(self.corpus[doc_id]['fact'])
        return query, pos, negs
    
    def get_pos_pairs(self):
        for q_id in self.query.keys():
            for doc_id in self.label[q_id]:
                if self.label[q_id][doc_id] >  1:  # gold label: 2, 3
                    self.pos_pairs.append((q_id, doc_id))

    def load_data(self, args: DataArguments):
        if args.train_query_path.endswith(".pkl"):
            with open(args.train_query_path, "rb") as f:
                self.query = pickle.load(f)
        else:
            raise ValueError(f"train_query_path {args.train_query_path} is not supported.")
        if args.corpus_path.endswith(".pkl"):
            with open(args.corpus_path, "rb") as f:
                self.corpus = pickle.load(f)
        else:
            raise ValueError(f"corpus_path {args.corpus_path} is not supported.")
        if args.label_path.endswith(".pkl"):
            with open(args.label_path, "rb") as f:
                self.label = pickle.load(f)
        else:
            raise ValueError(f"label_path {args.label_path} is not supported.")
        if args.hard_negative_path.endswith(".pkl"):
            with open(args.hard_negative_path, "rb") as f:
                self.hard_negative = pickle.load(f)
        else:
            raise ValueError(f"hard_negative_path {args.hard_negative_path} is not supported.")
            


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    继承了DataCollatorWithPadding的全部方法
    return: Tuple[BatchEncoding, BatchEncoding, List[BatchEncoding]]
            with shape of (batch_size, seq_len), (batch_size, seq_len), List[(negs_num, seq_len)] of bs length
    """
    query_max_len = 512
    doc_max_len = 512

    def __call__(self, features):
        queries, pos, negs = zip(*features)  # queries: List[str], pos: List[str], negs: List[List[str]]

        q_collated = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt"
        )
        pos_collated = self.tokenizer(
            pos,
            padding=True,
            truncation=True,
            max_length=self.doc_max_len,
            return_tensors="pt"
        )

        negs_collated = []
        for neg in negs:
            negs_collated.append(self.tokenizer(
                neg,
                padding=True,
                truncation=True,
                max_length=self.doc_max_len,
                return_tensors="pt"
            )
        )
        return q_collated, pos_collated, negs_collated
