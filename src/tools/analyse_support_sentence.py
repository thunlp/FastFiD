import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import json
import pandas as pd
from tasks.dense_retriever.metrics import has_answer
from tasks.dense_retriever.tokenizers import SimpleTokenizer
from tasks.reader.eval_utils import exact_match_score, metric_max_over_ground_truths
from tqdm import tqdm
import math


if __name__ == "__main__":
    support_sentence_file = "./checkpoints/hybrid_top25_constantlr/prediction_top25_support_sentence_epoch9/nq-test-extraction.json"
    support_sentences = json.load(open(support_sentence_file))
    prediction_file = "./checkpoints/hybrid_top25_constantlr/prediction_top25_generate_epoch9/nq-test.csv"
    predictions = pd.read_csv(prediction_file)
    tokenizer = SimpleTokenizer()
    top_k_hits_correct = [0] * len(support_sentences[0]['predict_support_sentences'])
    correct_number = 0
    top_k_hits_wrong = [0] * len(support_sentences[0]['predict_support_sentences'])
    wrong_number = 0
    top_k_hits = [0] * len(support_sentences[0]['predict_support_sentences'])
    for index in tqdm(predictions.index):
        p = predictions.loc[index, 'prediction']
        answers = eval(predictions.loc[index, 'answers'])
        if isinstance(p, float) and math.isnan(p):
            p = ''
        em = metric_max_over_ground_truths(exact_match_score, p, answers)
        sentences = support_sentences[index]['predict_support_sentences']
        sentence_hits = []
        for s in sentences:
            # print(s, p)
            sentence_hits.append(has_answer([p], s['prediction'], tokenizer, "string"))
        # sentence_hits = [has_answer([p], s['prediction'], tokenizer, "string") for s in sentences]
        best_hit = next((i for i, x in enumerate(sentence_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        if em:
            correct_number += 1
            if best_hit is not None:
                top_k_hits_correct[best_hit:] = [v + 1 for v in top_k_hits_correct[best_hit:]]
        else:
            wrong_number += 1
            if best_hit is not None:
                top_k_hits_wrong[best_hit:] = [v + 1 for v in top_k_hits_wrong[best_hit:]]
    topk_accuracies = [1, 5, 10, 20]
    for k in topk_accuracies:
        print("Recall@{}:".format(k), top_k_hits[k - 1] / len(support_sentences))
    print("Only EM = 1 Instances:")
    for k in topk_accuracies:
        print("Recall@{}:".format(k), top_k_hits_correct[k - 1] / correct_number)
    print("Only EM = 0 Instances:")
    for k in topk_accuracies:
        print("Recall@{}:".format(k), top_k_hits_wrong[k - 1] / wrong_number)
    
        

