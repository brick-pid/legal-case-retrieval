from transformers.trainer import *
from sentence_transformers import SentenceTransformer, models


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)

class BiTrainer(Trainer):
    def _save(self, output_dir: Optional[str]=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, "save"):
            raise NotImplementedError(f'{self.model.__class__} does not implement save method.')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():  # 主进程保存tokenizer
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers
        if self.is_world_process_zero():
            save_ckpt_for_sentence_transformers(output_dir, self.args.sentence_pooling_method, self.args.normalized)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写 Trainer 的 compute_loss 方法
        """
        query, pos, negs = inputs
        outputs = model(query, pos, negs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss