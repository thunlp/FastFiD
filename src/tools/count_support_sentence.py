import json
import sys

# argv1: filepath for data with sentence annotation

if __name__ == "__main__":
    filepath = sys.argv[1]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    support_sentence_num = []
    for d in data:
        contexts = d['ctxs']
        sentence_num = 0
        for context in contexts:
            assert len(context["start"]) == len(context["end"])
            sentence_num += len(context["start"])
            
        support_sentence_num.append(sentence_num)
    # print(support_sentence_num)
    print("Mean:", sum(support_sentence_num) / len(support_sentence_num))
    support_sentence_num.sort()
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for ratio in ratios:
        index = min(int(len(support_sentence_num) * ratio), len(support_sentence_num) - 1)
        print(f"{ratio * 100}% support sentence num:", support_sentence_num[index])