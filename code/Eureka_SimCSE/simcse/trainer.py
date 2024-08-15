import collections
import inspect
import math
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
import huggingface_hub.utils as hf_hub_utils
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    enable_full_determinism,
    find_executable_batch_size,
    set_seed,
    speed_metrics,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import (
    WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_apex_available,
    is_datasets_available,
    is_peft_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator

import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast, GradScaler

if is_datasets_available():
    import datasets

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

# DeepSpeed integration
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

from transformers.optimization import Adafactor, AdamW, get_scheduler
import copy

from datasets import load_from_disk
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, cast
from torch.utils.data.dataset import Dataset
import json
from tqdm.auto import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

TRAINER_STATE_NAME = "trainer_state.json"

import numpy as np
from datetime import datetime
from filelock import FileLock

import optuna

logger = logging.get_logger(__name__)


from torch import Tensor

def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
import faiss

def faiss_cosine_sim(q_vectors, training_vectors, topN, d):
    if isinstance(q_vectors, torch.Tensor):
        q_vectors = q_vectors.numpy()
    if isinstance(training_vectors, torch.Tensor):
        training_vectors = training_vectors.numpy()

    # Normalize training_vectors
    q_norms = np.linalg.norm(q_vectors, ord=2, axis=-1, keepdims=True)
    q_vectors = q_vectors / q_norms

    training_norms = np.linalg.norm(training_vectors, ord=2, axis=-1, keepdims=True)
    training_vectors = training_vectors / training_norms

    index = faiss.IndexFlatIP(d)  # the other index，需要以其他index作为基础

    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.train(training_vectors)
    index.add(training_vectors)
    D, I = index.search(q_vectors, topN)
    score_list = D.tolist()
    index_list = I.tolist()

    return index_list, score_list

def batch_faiss_cosine_sim(q_embed, doc_emb, d, batch_size=100, topN=200):
    all_index_l = []
    all_score_l = []
    num_queries = len(q_embed)

    for i in range(0, num_queries, batch_size):
        batch_q_embed = q_embed[i:i + batch_size]
        index_l, score_l = faiss_cosine_sim(batch_q_embed, doc_emb, topN=topN, d=d)
        all_index_l.extend(index_l)
        all_score_l.extend(score_l)

    return all_index_l, all_score_l

from prettytable import PrettyTable
def print_table(task_names, metrics_name, scores):
    metrics_name = [''] + metrics_name
    tb = PrettyTable()
    tb.field_names = metrics_name
    for m,s in zip(task_names,scores):
        tb.add_row([m]+s)
    print(tb)

def get_hit_rate(pos,recall):

    total = (len(pos))

    hit1, hit5, hit10, hit20, hit50, hit100, hit200, hit500 = 0., 0., 0., 0., 0., 0., 0., 0.

    for p,r in zip(pos,recall):

        golden = set(p)

        if golden.issubset(set(r[:1])):
            hit1 += 1
        if golden.issubset(set(r[:5])):
            hit5 += 1
        if golden.issubset(set(r[:10])):
            hit10 += 1 
        if golden.issubset(set(r[:20])):
            hit20 += 1 
        if golden.issubset(set(r[:50])):
            hit50 += 1 
        if golden.issubset(set(r[:100])):
            hit100 += 1 
        if golden.issubset(set(r[:200])):
            hit200 += 1 
        if golden.issubset(set(r[:500])):
            hit500 += 1 

    return {
        'hit1': hit1,
        'hit5': hit5, 
        'hit10': hit10, 
        'hit20': hit20, 
        'hit50': hit50, 
        'hit100': hit100, 
        'hit200': hit200, 
        'hit500': hit500,
        'total': total
    }

class CLTrainer(Trainer):
    def __init__(self,
        corpus_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus_dataset = corpus_dataset


    @torch.no_grad()
    def encode(self,
            sentences: Union[List[str], str],
            batch_size: int = 512,
            max_length: int = 512,
            pooler_type: str = 'cls',
            convert_to_numpy: bool = True) -> np.ndarray:
        
        self.num_gpus = torch.cuda.device_count()
        self.device = self.model.device
        self.normalize_embeddings = True

        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus

        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < batch_size):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                pad_to_multiple_of=8,
            ).to(self.device)

            if pooler_type == 'cls_before_pooler':
                last_hidden_state = self.model(**inputs, sent_emb=True, return_dict=True).last_hidden_state
                embeddings = last_hidden_state[:,0,:]
            elif pooler_type == 'avg':
                last_hidden_state = self.model(**inputs,  sent_emb=True, return_dict=True).last_hidden_state
                embeddings = last_token_pool(last_hidden_state, inputs['attention_mask'])
            else:
                embeddings = self.model(**inputs, output_hidden_states=True, sent_emb=True, return_dict=True).pooler_output

            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def evaluate(
        self,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        pooler_type: str = 'cls',
    ) -> Dict[str, float]:

        corpus_dict = json.loads(self.corpus_dataset['task2corpus'][0])
        tasks = list(corpus_dict)
        
        task_data = {}
        for t in tasks:
            task_data[t] = []
        
        for item in self.eval_dataset:
            task_data[item['task_type']].append(item['item'])

        metrics_name = ['HR@1', 'HR@5', 'HR@10', 'HR@20', 'HR@50', 'HR@100', 'HR@200', 'HR@500']
        results = {}
        eval_result = {}

        for t in tasks:

            document_list = corpus_dict[t]
            if isinstance(self.model, SentenceTransformer):
                doc_emb = self.model.encode(document_list)
            else:
                doc_emb = self.encode(document_list,pooler_type=pooler_type)

            t_data = task_data[t]
            queries = [i[0] for i in t_data]
            pos_list = [i[1].split('<#SEP#>') for i in t_data]

            if isinstance(self.model, SentenceTransformer):
                q_embed = self.model.encode(queries)
            else:
                q_embed = self.encode(queries,pooler_type=pooler_type)

            index_l, score_l = batch_faiss_cosine_sim(q_embed, doc_emb, topN=200, d=len(doc_emb[0]))

            recall_result = [[document_list[i] for i in j] for j in index_l]

            task_eval_result = get_hit_rate(pos_list,recall_result)

            eval_result[t] = task_eval_result

            del doc_emb
            del q_embed
            torch.cuda.empty_cache()

        task_map = {
            'Clinical_examination_zh': 'Term_zh',
            'Disease_dignosis_zh': 'Term_zh',
            'Procedure_operation_zh': 'Term_zh',
            'Symptom_sign_zh': 'Term_zh',
            'Clinical_examination_cross': 'Term_cross',
            'Disease_dignosis_cross': 'Term_cross',
            'Procedure_operation_cross': 'Term_cross',
            'Symptom_sign_cross': 'Term_cross',
            
        }
        
        merge_task = list(set([v for k,v in task_map.items() if k in eval_result]))

        merge_task_result = {k:{} for k in merge_task}

        fix_eval_result = {}

        for k in eval_result:
            if k not in task_map:
                fix_eval_result[k] = eval_result[k]
            else:
                for item in eval_result[k]:
                    if item in merge_task_result[task_map[k]]:
                        merge_task_result[task_map[k]][item] += eval_result[k][item]
                    else:
                        merge_task_result[task_map[k]][item] = eval_result[k][item]
                        
        for k in merge_task_result:
            fix_eval_result[k] = merge_task_result[k]

        task_names = list(fix_eval_result)

        for t in task_names:
            hit_rates = {}
            for k in fix_eval_result[t]:
                if 'hit' in k:
                    hit_rates['HR@'+k.replace('hit','')] = fix_eval_result[t][k] / fix_eval_result[t]['total']
            results[t] = hit_rates

        task_scores = []
        for task in task_names:
            if task in results:
                format_scores = [results[task][metric] for metric in metrics_name]
                task_scores.append(format_scores)
            else:
                format_scores = [0.00] * len(metrics_name)
                task_scores.append(format_scores)

        task_names.append("Avg.")
        task_avg_scores = [sum([float(task_scores[i][j]) for i in range(len(task_scores))]) / (len(task_names)-1) for j in range(len(metrics_name))]
        task_scores.append(task_avg_scores)
        # print_table(task_names, metrics_name, task_scores)
        metrics = {metric_key_prefix+'_'+metric: avg_s for metric,avg_s in zip(metrics_name,task_avg_scores)}

        self.log(metrics)
        return metrics

    def _save_checkpoint(self, model, trial, metrics=None):
        # assert  (model) is self.model, "internal model should be a reference to self.model"

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                output_dir = self.args.output_dir
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                self.save_model(output_dir)
                if self.deepspeed:
                    self.deepspeed.save_checkpoint(output_dir)

                if is_torch_tpu_available():
                    xm.rendezvous("saving_optimizer_states")
                    xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        reissue_pt_warnings(caught_warnings)
                elif self.is_world_process_zero() and not self.deepspeed:
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)

                if self.is_world_process_zero():
                    self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        else:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is not None and trial is not None:
                if self.hp_search_backend == HPSearchBackend.OPTUNA:
                    run_id = trial.number
                else:
                    from ray import tune

                    run_id = tune.get_trial_id()
                run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
                output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
            else:
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                self.store_flos()

            self.save_model(output_dir)
            if self.deepspeed:
                self.deepspeed.save_checkpoint(output_dir)

            if is_torch_tpu_available():
                xm.rendezvous("saving_optimizer_states")
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)
            elif self.is_world_process_zero() and not self.deepspeed:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)

            if self.is_world_process_zero():
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            if self.is_world_process_zero():
                self._rotate_checkpoints(use_mtime=True)

    def _save_checkpoint(self, model, trial, metrics=None):

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            # Update the `TrainerControl` state to where we are currently
            self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)


    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None,ignore_keys_for_eval: Optional[List[str]] = None, **kwargs):
        self._hp_search_setup(trial)

        model_reloaded = False
        if self.model_init is not None:
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            self.optimizer, self.lr_scheduler = None, None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, self.args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, self.args.auto_find_batch_size
        )
        if self.args.push_to_hub:
            try:
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=self.args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=self.args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
