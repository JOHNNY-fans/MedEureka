import logging
import torch
import torch
import faiss
import faiss.contrib.torch_utils
logger = logging.getLogger(__name__)
import random
import numpy as np
from tqdm import tqdm
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, cast
from torch import Tensor
from transformers import Trainer,AutoModel
from dataclasses import dataclass, field
import json
from torch.utils.data.dataset import Dataset
from prettytable import PrettyTable
import faiss
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel
from FlagEmbedding import FlagLLMModel
from FlagEmbedding import BGEM3FlagModel

def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
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

def batch_faiss_cosine_sim(q_embed, doc_emb, d, batch_size=200, topN=200):
    all_index_l = []
    all_score_l = []
    num_queries = len(q_embed)

    for i in range(0, num_queries, batch_size):
        batch_q_embed = q_embed[i:i + batch_size]
        index_l, score_l = faiss_cosine_sim(batch_q_embed, doc_emb, topN=topN, d=d)
        all_index_l.extend(index_l)
        all_score_l.extend(score_l)

    return all_index_l, all_score_l

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

def get_detailed_instruct(task_description: str) -> str:
    return f'Instruct: {task_description}\nQuery: '

query_prompt_dict = {}
query_prompt_dict['Study_en'] = 'Given a query for the experts, retrieve relevant scientific articles.'
query_prompt_dict['Study_zh'] = '给定一个来自患者的问题，查询相关的医学片段。'
query_prompt_dict['Table_en'] = 'Given a query, retrieve the relevant medical table.'
query_prompt_dict['Table_zh'] = '给定一个问题，查询相关的医学诊疗表格。'
query_prompt_dict['KB_para_zh'] = '给定一个问题，查询中药药品说明书中的相关片段。'
query_prompt_dict['KB_doc_zh'] = '给定一个问题，查询相关的中药药品说明书。'
query_prompt_dict['AskAPatient_en'] = 'Given a social media phrase, retrieve relevant medical terminology'
query_prompt_dict['SMM4H-17_en'] = 'Given a social media phrase, retrieve relevant medical terminology'
query_prompt_dict['TwADR-L_en'] = 'Given a social media phrase, retrieve relevant medical terminology'
query_prompt_dict['Disease_dignosis_zh'] = '给定一个短语，查询标准的疾病诊断术语。'
query_prompt_dict['Clinical_examination_zh'] = '给定一个短语，查询标准的体格检查短语。'
query_prompt_dict['Procedure_operation_zh'] = '给定一个短语，查询标准的手术操作短语。'
query_prompt_dict['Symptom_sign_zh'] = '给定一个短语，查询标准的症状体征短语。'
query_prompt_dict['Disease_dignosis_cross'] = 'Given a phrase, retrieve normalized disease diagnosis term.'
query_prompt_dict['Clinical_examination_cross'] = 'Given a phrase, retrieve normalized clinical examination term.'
query_prompt_dict['Procedure_operation_cross'] = 'Given a phrase, retrieve normalized procedure operation term.'
query_prompt_dict['Symptom_sign_cross'] = 'Given a phrase, retrieve normalized symptom sign term.'
query_prompt_dict['EHR_query2sql_zh'] = '给定一个关于医疗电子病历的问题，查询相关的SQL语句。'#给定一个查询特定患者的问题，查询相关的SQL语句
query_prompt_dict['EHR_sql2para_zh'] = '给定一个SQL语句，查询相关的医疗电子病历段落。'#给定一个SQL语句，查询相关的电子病历片段
query_prompt_dict['EHR_query2para_zh'] = '给定一个问题，查询相关的医疗电子病历段落。'
query_prompt_dict['EHR_query2doc_zh'] = '给定一个问题，查询相关的医疗电子病历。'#给定一个问题，查询相关的电子病历文档
query_prompt_dict['Dialogue_qnorm_zh'] = '给定一个来自患者的问题，查询相关的问题。'
query_prompt_dict['Dialogue_en'] = 'Given a query for patients, retrieve relevant medical patient\'s questions'
query_prompt_dict['Dialogue_zh'] = '给定一个来自患者的问题，查询相关的回答。'

class CLTrainer(Trainer):

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
                last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
                embeddings = last_hidden_state[:,0,:]
            elif pooler_type == 'avg':
                last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
                embeddings = last_token_pool(last_hidden_state, inputs['attention_mask'])
            else:
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
                
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
        model: SentenceTransformer | FlagLLMModel | FlagModel | BGEM3FlagModel | CLTrainer ,
        eval_dataset: Optional[Dataset] = None,
        corpus_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        pooler_type: str = 'cls',
        eval_senteval_transfer: bool = False,
        need_prompt_for_task: bool = False,
        model_type: str = 'FlagLLMModel',
        emb_batch_size: int = 16,
        q_batch_size: int = 200,
    ) -> Dict[str, float]:

        corpus_dict = json.loads(corpus_dataset['task2corpus'][0])
        tasks = list(corpus_dict)
        
        task_data = {}
        for t in tasks:
            task_data[t] = []
        
        for item in eval_dataset:
            task_data[item['task_type']].append(item['item'])

        metrics_name = ['HR@1', 'HR@5', 'HR@10', 'HR@20', 'HR@50', 'HR@100', 'HR@200', 'HR@500']
        results = {}
        eval_result = {}

        for t in tasks:

            document_list = corpus_dict[t]
            t_data = task_data[t]
            queries = [i[0] for i in t_data]
            pos_list = [i[1].split('<#SEP#>') for i in t_data]

            if isinstance(model, SentenceTransformer):
                doc_emb = model.encode(document_list,batch_size=emb_batch_size)
                if need_prompt_for_task: 
                    q_embed = model.encode(queries, prompt= get_detailed_instruct(query_prompt_dict[t]))
                else: q_embed = model.encode(queries)

            elif model_type == 'FlagLLMModel' or model_type == 'FlagModel':
                if need_prompt_for_task: model.query_instruction_for_retrieval = query_prompt_dict[t]
                doc_emb = model.encode_corpus(document_list,batch_size=emb_batch_size)
                q_embed = model.encode_queries(queries)
            
            elif model_type == 'bge':
                doc_emb = np.array(model.encode(document_list,batch_size=emb_batch_size)['dense_vecs']).astype(np.float16)
                q_embed = np.array(model.encode(queries)['dense_vecs']).astype(np.float16)

            else:
                doc_emb = CLTrainer.encode(document_list,pooler_type=pooler_type)


            index_l, score_l = batch_faiss_cosine_sim(q_embed, doc_emb, batch_size=q_batch_size, topN=500, d=len(doc_emb[0]))

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
                format_scores = ["%.4f" % results[task][metric] for metric in metrics_name]
                task_scores.append(format_scores)
            else:
                format_scores = ["0.00"] * len(metrics_name)
                task_scores.append(format_scores)

        task_names.append("Avg.")
        task_avg_scores = ["%.4f" % (sum([float(task_scores[i][j]) for i in range(len(task_scores))]) / (len(task_names)-1) ) for j in range(len(metrics_name))]
        task_scores.append(task_avg_scores)

        print_table(task_names, metrics_name, task_scores)

        metrics = {metric: avg_s for metric,avg_s in zip(metrics_name,task_avg_scores)}
        return metrics