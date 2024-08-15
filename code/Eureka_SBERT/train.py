import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict

import json

from datasets import load_dataset, load_from_disk
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sbert.trainer import MyTrainer
from sentence_transformers.trainer import SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss, SoftmaxLoss, CoSENTLoss
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: str = field(
        default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    use_auth_token: bool = field(
        default=False, metadata={"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script with private models)."}
    )

@dataclass
class DataTrainingArguments:
    dataset_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=32, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."}
    )
    pad_to_max_length: bool = field(
        default=False, metadata={"help": "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5, metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"}
    )

@dataclass
class OurTrainingArguments(SentenceTransformerTrainingArguments):
    eval_transfer: bool = field(
        default=False, metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    model = SentenceTransformer(model_args.model_name_or_path)

    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f"Making {name} contiguous")
            param.data = param.data.contiguous()
            # 验证连续性
            if not param.is_contiguous():
                print(f"Failed to make {name} contiguous")

    dataset = load_from_disk(data_args.dataset_name_or_path)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    corpus_dataset = dataset['corpus']

    convert_train_dict = {
        "anchor": train_dataset["query"],
        "positive": train_dataset["pos"],
        "negative": train_dataset["neg"]
    }

    convert_train_dataset = Dataset.from_dict(convert_train_dict)

    mnrl_loss = MultipleNegativesRankingLoss(model)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=convert_train_dataset,
        eval_dataset=eval_dataset,
        corpus_dataset=corpus_dataset,
        loss=mnrl_loss,
    )

    if training_args.do_train:
        trainer.train()
        with open(os.path.join(training_args.output_dir, "trainer_state.json"), "w") as f:
                json.dump(trainer.state.__dict__, f, indent=4)
def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()
