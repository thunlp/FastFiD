import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import json
from copy import deepcopy
from tqdm import tqdm
from data.orqa_wiki_dataset import OpenRetrievalEvidenceDataset
import argparse

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file-train", type=str, default=None, help="train file to convert.")
    parser.add_argument("--qa-file-dev", type=str, default=None, help="dev file to convert.")
    parser.add_argument("--qa-file-test", type=str, default=None, help="test file to convert.")
    parser.add_argument("--output-dir", type=str, default=None, help="output dir.")
    parser.add_argument("--wikipedia-path", type=str, default="datas/psgs_w100.tsv", help='wikipedia passage path')
    parser.add_argument("--title-answer", action='store_true', help="Whether to use title() for answers in train. Used on TriviaQA.")
    return parser

def title_answer(answer):
    if answer.isupper():
        answer = answer.title()
    return answer

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    data_paths = {
        'train': args.qa_file_train,
        'dev': args.qa_file_dev,
        'test': args.qa_file_test,
    }
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print("loading wikipedia data ...")
    samples, id2text = OpenRetrievalEvidenceDataset.process_samples_from_single_path(args.wikipedia_path)
    print(f"load wikipedia data over. Total {len(samples)} passages.")
    for split in ['dev', 'test', 'train']:
        data_path = data_paths[split]
        if data_path is None:
            continue
        with open(data_path, 'r') as f:
            data = json.load(f)
        new_datas = []
        for d in tqdm(data, desc=split):
            new_d = deepcopy(d)
            if args.title_answer and split == 'train':
                answers = [title_answer(a) for a in new_d['answers']]
                new_d['answers'] = answers
            for ctx_index, ctx in enumerate(d['ctxs']):
                row_id = int(ctx['id'])
                text = id2text[row_id][0]
                title = id2text[row_id][1]
                new_d['ctxs'][ctx_index]['title'] = title
                new_d['ctxs'][ctx_index]['text'] = text
            new_datas.append(new_d)
        filename = data_path.split('/')[-1]
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(new_datas, f, ensure_ascii=False, indent=4)
        

        