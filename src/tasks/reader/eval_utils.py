# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Evaluation utilities."""
import re
import string
import collections
from tqdm import tqdm

import unicodedata

rouge = None


def normalize_answer(s):
    """Normalize answer."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def rouge_match_score(prediction, ground_truth):
    global rouge
    import evaluate
    if rouge is None:
        rouge = evaluate.load('rouge')
    rouge_score = rouge.compute(predictions=[prediction], references=[[ground_truth]])
    return rouge_score["rougeL"]

def answer_recall_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)

def answer_strem_score(prediction, ground_truth):
    n_short_answers = [normalize_answer(sa) for sa in ground_truth]
    n_context = normalize_answer(prediction)

    for ans in n_short_answers:
        if ans in n_context:
            return True
    return False

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, ground_truth):
    try:
        regex = re.compile(ground_truth,
                           flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
        return regex.match(prediction) is not None
    except re.error:
        return False


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)

def metric_mean_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    return sum(scores_for_ground_truths) / len(scores_for_ground_truths)

def postprocess(result, max_answer_length, tokenizer, use_sum=True, topk=10):
    """Post process extractive predictions"""
    predictions = []
    
    def default_value():
        return 0

    its_input_ids = result.input_ids
    scores = collections.defaultdict(default_value)
    for topi, topi_input_ids in enumerate(its_input_ids):
        selected_prob = result.selected_probs[topi][1]
        sequence_len = result.sequence_lengths[topi]
        start_probs = result.start_probs[topi][:sequence_len]
        end_probs = result.end_probs[topi][:sequence_len]
        for (i, s) in enumerate(start_probs):
            for (j, e) in enumerate(end_probs[i:i+max_answer_length]):
                if use_sum:
                    span = tokenizer.decode(topi_input_ids[i:i + j + 1], skip_special_tokens=True)
                else:
                    span = (topi, i, j)
                scores[span] += (s * e * selected_prob)
    scores_items = list(scores.items())
    scores_items.sort(key=lambda p:p[1], reverse=True)
    for index in range(len(scores_items)):
        if use_sum:
            prediction = scores_items[index][0]
            doc_index = -1
            predictions.append({'prediction': prediction})
        else:
            doc_index, i, j = scores_items[index][0]
            score = scores_items[index][1]
            prediction = tokenizer.decode(its_input_ids[doc_index][i:i + j + 1], skip_special_tokens=True)
            predictions.append({'prediction': prediction, 'doc_index': doc_index, 'start': i, 'end': i + j, 'score': score})
        if len(predictions) == topk:
            break
    return predictions

from tasks.dense_retriever.metrics import has_answer
from tasks.dense_retriever.tokenizers import SimpleTokenizer
def extraction_recall_score(predictions, references, match_type):
    tokenizer = SimpleTokenizer()
    hits = [has_answer(references, p['prediction'], tokenizer, match_type) for p in predictions]
    top_k_hits = [0] * len(predictions)
    best_hit = next((i for i, x in enumerate(hits) if x), None)
    if best_hit is not None:
        top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    return top_k_hits

if __name__ == "__main__":
    ground_truths = [
            [
                "Daei",
                "Ali Daei"
            ],
            [
                "Bican",
                "Josef Bican"
            ],
            [
                "Sinclair",
                "Christine Sinclair"
            ]
        ]
    prediction = "Ali Dael has the highest goals in men's world international football with 109 goals. Josef Bicans has the highest goals all-time in men's football and Christine Sinclairs has the highest goals in women's world international football."
    print(metric_mean_over_ground_truths(answer_strem_score, prediction, ground_truths))
