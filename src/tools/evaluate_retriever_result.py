import json
import tqdm


if __name__ == "__main__":
    dev_file = "./data/qaps_text/nq-test.json"
    topk_list = [1, 5, 10, 20, 25, 50, 100]
    topk = max(topk_list)
    with open(dev_file) as f:
        dev_result = json.load(f)
        total_length = 2000
        have_answers = [0 for _ in range(topk)]
        for i in tqdm.trange(total_length):
            result = dev_result[i]
            for rank in range(topk):
                ctx = result['ctxs'][rank]
                if ctx['has_answer']:
                    if 'start' in ctx:
                        assert len(ctx['start']) > 0
                    for j in range(rank, topk):
                        have_answers[j] += 1
                    break
                else:
                    if 'start' in ctx:
                        assert len(ctx['start']) == 0
        for k in topk_list:
            print("Top-{}: {:.4f}".format(k, have_answers[k - 1] / total_length))