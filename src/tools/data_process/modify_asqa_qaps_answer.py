import json
import pandas as pd

train_qaps_filepath = "./datas/qaps_fidkd_text_sentence/asqa-train.json"
dev_qaps_filepath = "./datas/qaps_fidkd_text_sentence/asqa-dev.json"

train_filepath = "./datas/asqa/data/train-00000-of-00001-87b7d64f7913b544.parquet"
dev_filepath = "./datas/asqa/data/dev-00000-of-00001-58a9a40c6e69f07b.parquet"

output_train_path = "./datas/qaps_fidkd_text_sentence/asqa-train-2.json"
output_dev_path = "./datas/qaps_fidkd_text_sentence/asqa-dev-2.json"


def modify_short_answer(qaps_path, parquet_path, output_path, split):
    print("Modify short answers in {} ...".format(split))
    qaps = json.load(open(qaps_path, 'r', encoding='utf-8'))
    data = pd.read_parquet(parquet_path)
    for i, row in data.iterrows():
        qap_data = qaps[i]
        short_answers = []
        for qa_pair in row['qa_pairs']:
            short_answers.append(qa_pair['short_answers'].tolist())
        answers = qap_data['answers']
        short_answer_num = sum([len(sa) for sa in short_answers])
        assert len(answers) == short_answer_num, f"Index {i} error"
        qap_data['answers'] = short_answers
    print("process {} datas".format(len(qaps)))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qaps, f, indent=4, ensure_ascii=False)
    print("dump {} data to {}".format(split, output_path))

if __name__ == "__main__":
    modify_short_answer(train_qaps_filepath, train_filepath, output_train_path, "train")
    modify_short_answer(dev_qaps_filepath, dev_filepath, output_dev_path, "dev")