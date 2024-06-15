import pandas as pd

train_filepath = "./datas/asqa/train-00000-of-00001-87b7d64f7913b544.parquet"
dev_filepath = "./datas/asqa/dev-00000-of-00001-58a9a40c6e69f07b.parquet"
output_train_path = "./datas/qas/asqa-train.csv"
output_dev_path = "./datas/qas/asqa-dev.csv"

if __name__ == "__main__":
    print("Train:")
    train_data = pd.read_parquet(train_filepath)
    new_train_data = []
    train_cnt = 0
    for index, row in train_data.iterrows():
        question = row['ambiguous_question']
        short_answers = []
        for t in row['qa_pairs']:
            for a in t['short_answers']:
                short_answers.append(a.strip())
        long_answers = [t['long_answer'].strip() for t in row['annotations']]
        new_train_data.append({
            'question': question,
            'answers': short_answers,
            'long_answers': long_answers,
        })
    with open(output_train_path, 'w', encoding='utf-8') as f:
        for d in new_train_data:
            f.write("{}\t{}\t{}\n".format(d["question"], str(d["answers"]), str(d["long_answers"])))
    # print(f"Load {len(new_train_data)} datas.")
    
    print("Dev:")
    dev_data = pd.read_parquet(dev_filepath)
    new_dev_data = []
    dev_cnt = 0
    for index, row in dev_data.iterrows():
        question = row['ambiguous_question']
        short_answers = []
        for t in row['qa_pairs']:
            for a in t['short_answers']:
                short_answers.append(a.strip())
        long_answers = [t['long_answer'].strip() for t in row['annotations']]
        new_dev_data.append({
            'question': question,
            'answers': short_answers,
            'long_answers': long_answers,
        })
        # if index == 6:
        #     print(row['qa_pairs'])
        #     quit()
    with open(output_dev_path, 'w', encoding='utf-8') as f:
        for d in new_dev_data:
            f.write("{}\t{}\t{}\n".format(d["question"], str(d["answers"]), str(d["long_answers"])))
    print(f"Load {len(new_dev_data)} datas.")