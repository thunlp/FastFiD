import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from tasks.dense_retriever.tokenizers import SimpleTokenizer
from spacy.lang.en import English
from tqdm import tqdm
import argparse
from functools import partial

nlp = None
simple_tokenizer = SimpleTokenizer()

def init_spacy():
    global nlp
    if nlp is None:
        nlp = English()
        nlp.add_pipe("sentencizer")

def mark_begin_end_sentence_in_text(answer, text):
    global nlp
    init_spacy()
    assert nlp is not None
    doc = nlp(text)
    length = 0
    sentences = []
    sentence_starts = []
    sentence_ends = []
    begin_index = 0
    span_starts, span_ends, _ = mark_begin_end_span_in_text(answer, text)
    # print(span_starts, span_ends)
    for i, sent in enumerate(doc.sents):
        sent = str(sent)
        sentences.append(sent)
        sentence_start = text.index(sent, begin_index)
        sentence_starts.append(sentence_start)
        sentence_ends.append(sentence_start + len(sent) - 1)
        begin_index = sentence_start + len(sent)
        # starts, ends, _ = mark_begin_end_span_in_text(answer, sent)
        # have_answers.append(len(starts))
        # print(sentence_start, sent)
    span_index = 0
    starts = []
    ends = []
    span_cross = False
    for span_start, span_end in zip(span_starts, span_ends):
        for sent, sentence_start, sentence_end in zip(sentences, sentence_starts, sentence_ends):
            if span_start >= sentence_start and span_end <= sentence_end:
                starts.append(sentence_start)
                ends.append(sentence_end)
            if span_start >= sentence_start and span_end > sentence_end and span_start <= sentence_end:
                starts.append(sentence_start)
            if span_start < sentence_start and span_end <= sentence_end and span_end >= sentence_start:
                ends.append(sentence_end)
            if span_start < sentence_start and span_end > sentence_end:
                continue
    if len(starts) != 0:
        starts_ends = [(s, e) for s, e in zip(starts, ends)]
        # print(starts_ends)
        starts_ends.sort(key=lambda p: p[1] - p[0] + 1, reverse=True)
        # print(starts_ends)
        new_starts_ends = []
        for index, (s, e) in enumerate(starts_ends):
            be_included = False
            for i in range(index):
                i_s, i_e = starts_ends[i]
                if s >= i_s and e <= i_e:
                    be_included = True
            if not be_included:
                new_starts_ends.append((s, e))
        new_starts_ends.sort(key=lambda p: p[0])
        starts, ends = zip(*new_starts_ends)
    return starts, ends
         
def mark_begin_end_span_in_text(answer, text):
    tokenized_text = simple_tokenizer.tokenize(text)
    text_words = tokenized_text.words(uncased=True)
    text_offsets = tokenized_text.offsets()
    answer_words = simple_tokenizer.tokenize(answer).words(uncased=True)
    def subfinder(mylist, pattern):
        match_indexes = []
        for i in range(len(mylist) - len(pattern) + 1):
            if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
                match_indexes.append(i)
        return match_indexes
    match_indexes = subfinder(text_words, answer_words)
    start_positions = []
    end_positions = []
    for i in match_indexes:
        # print(text_words[i: i + len(answer_words)])
        start = text_offsets[i][0]
        end = text_offsets[i + len(answer_words) - 1][1]
        start_positions.append(start)
        end_positions.append(end - 1)
    return start_positions, end_positions, text


def preprocess_data(filepath):
    print("Preprocess {} ...".format(filepath))
    import ijson
    f = open(filepath, 'r', encoding='utf-8')
    # data = json.load(f)
    data = ijson.items(f, 'item')
    new_data = []
    for data_index, d in enumerate(tqdm(data)):
        # if data_index != 52032:
        #     continue
        question = d['question']
        answers = d['answers']
        contexts = d['ctxs']
        new_answers = []
        for a in answers:
            answer_words = simple_tokenizer.tokenize(a).words(uncased=True)
            if len(answer_words) > 0:
                new_answers.append(a)
        assert len(new_answers) > 0, f"Index {data_index} data in {filepath} has no answer."
        d['answers'] = new_answers
        answers = new_answers
        for c_index, context in enumerate(contexts):
            current_starts = []
            current_ends = []
            for a in answers:
                starts, ends = mark_begin_end_sentence_in_text(a, context['text'])
                current_starts.extend(starts)
                current_ends.extend(ends)
            if len(current_starts) != 0:
                starts_ends = [(s, e) for s, e in zip(current_starts, current_ends)]
                # print(starts_ends)
                starts_ends.sort(key=lambda p: p[1] - p[0] + 1, reverse=True)
                # print(starts_ends)
                new_starts_ends = []
                for index, (s, e) in enumerate(starts_ends):
                    be_included = False
                    for i in range(index):
                        i_s, i_e = starts_ends[i]
                        if s >= i_s and e <= i_e:
                            be_included = True
                    if not be_included:
                        new_starts_ends.append((s, e))
                new_starts_ends.sort(key=lambda p: p[0])
                current_starts, current_ends = zip(*new_starts_ends)
            context['start'] = list(map(int, current_starts))
            context['end'] = list(map(int, current_ends))
            context['score'] = float(context['score'])
        new_data.append(d)
            # if question == 'who is the original singer of i write sins not tragedies' and c_index == 1:
            #     print(context)
            # assert context['has_answer'] == (len(current_starts) > 0), "Question: {}, context {}".format(question, c_index)
    f.close()
    return new_data

def mark_rouge_for_each_sentence(text, answers, rouge):
    global nlp
    init_spacy()
    assert nlp is not None
    doc = nlp(text)
    length = 0
    sentences = []
    sentence_starts = []
    sentence_ends = []
    rougel_scores = []
    begin_index = 0
    for i, sent in enumerate(doc.sents):
        sent = str(sent)
        sentences.append(sent)
        sentence_start = text.index(sent, begin_index)
        sentence_starts.append(sentence_start)
        sentence_ends.append(sentence_start + len(sent) - 1)
        begin_index = sentence_start + len(sent)
        rouge_score = rouge.compute(predictions=[sent], references=[answers])
        rougel_scores.append(rouge_score["rougeL"])
    sentences = list(zip(sentence_starts, sentence_ends, rougel_scores))
    return sentences

    
def rouge_process_function(d, rouge, topk_retrieval):
    question = d['question']
    answers = d['answers']
    contexts = d['ctxs']
    # print("question:", question)
    # print("answer:", answers[0])
    answer_sents = []
    for a in answers:
        doc = nlp(a)
        for sent in doc.sents:
            sent = str(sent)
            answer_sents.append(sent)
    total_sentence_num = 0
    selected_sentence_num = 0
    for rank in range(topk_retrieval):
        context = contexts[rank]
        # print("Rank {}:".format(rank))
        marked_sentences = mark_rouge_for_each_sentence(context['text'], answer_sents, rouge)
        current_starts = []
        current_ends = []
        for s in marked_sentences:
            start, end, score = s
            if score > 0.15:
                current_starts.append(start)
                current_ends.append(end)
        context['start'] = list(map(float, current_starts))
        context['end'] = list(map(float, current_ends))
        context['score'] = float(context['score'])
        total_sentence_num += len(marked_sentences)
        selected_sentence_num += len(current_starts)
    return d, selected_sentence_num, total_sentence_num

def preprocess_data_by_rouge(filepath, process_num=1, topk_retrieval=100):
    import evaluate
    rouge = evaluate.load('rouge')
    print("Preprocess {} ...".format(filepath))
    import ijson
    f = open(filepath, 'r', encoding='utf-8')
    # data = json.load(f)
    data = ijson.items(f, 'item')
    new_data = []
    selected_ratio = []
    global nlp
    init_spacy()
    assert nlp is not None
    map_function = map
    if process_num > 1:
        import multiprocessing
        print(f"Build {process_num} processes to preprocess data ...")
        pool = multiprocessing.Pool(processes=process_num)
        map_function = partial(pool.imap, chunksize=2)
    for data_index, (d, selected_sentence_num, total_sentence_num) \
        in enumerate(tqdm(map_function(partial(rouge_process_function, rouge=rouge, topk_retrieval=topk_retrieval), data))):
        selected_ratio.append(selected_sentence_num / total_sentence_num)
        new_data.append(d)
    mean_selected_ratio = sum(selected_ratio) / len(selected_ratio)
    f.close()
    print("Selected ratio for each question: {}".format(mean_selected_ratio))
    return new_data 
    
        
                
        
def debug():
    # debug
    text = "2002 Gordon Wharmby Gordon Wharmby (6 November 1933 – 18 May 2002) was a British television actor. He 2002 was best known for the role  of Wesley Pegden on \"Last of the Summer Wine\".    He was born in Manchester, Lancashire, in 1933, and served in the Royal Air Force during his national service.Wharmby was 1.223 originally employed as a painter and decorator and had no formal training as an actor. He gained stage experience with Oldham Repertory Theatre and worked part-time as a jobbing actor.\tEarly television roles included bit-parts in programmes such as \"Bill Brand\" (1976) 2002, \"The One and Only Phyllis"
    # text = 'forgive [himself] if [he] made others as miserable as [he] made [himself]." Lanser argues that the same argument of devastation and misery can be said about the work of Edgar Allan Poe, but his work is still printed and studied by academics. "The Yellow Wallpaper" provided feminists the tools to interpret literature in different ways. Lanser argues that the short story was a "particularly congenial medium for such a re-vision . . . because the narrator herself engages in a form of feminist interpretation when she tries to read the paper on her wall". The narrator in the story is'
    title = "Gordon Wharmby"
    question = "when did wesley leave last of the summer wine"
    answer = "2002"
    starts, ends = mark_begin_end_sentence_in_text(answer, text)
    print(starts, ends)
    print(len(text))
    for s, e in zip(starts, ends):
        print(text[s:e + 1])
        print('"' + text[e] + '"')

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file-train", type=str, default=None, help="train file to convert.")
    parser.add_argument("--qa-file-dev", type=str, default=None, help="dev file to convert.")
    parser.add_argument("--qa-file-test", type=str, default=None, help="test file to convert.")
    parser.add_argument("--output-dir", type=str, default=None, help="output dir.")
    parser.add_argument("--process-num", type=int, default=1, help="process num.")
    parser.add_argument("--topk-retrieval", type=int, default=100, help="process num.")
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    # debug()
    data_paths = {
        'train': args.qa_file_train,
        'dev': args.qa_file_dev,
        'test': args.qa_file_test,
    }
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ['train', 'test', 'dev']:
        filepath = data_paths[split_name]
        if filepath is None:
            continue
        if "eli5" in filepath:
            print("ELI5 dataset, use rouge to preprocess")
            data = preprocess_data_by_rouge(filepath, args.process_num, args.topk_retrieval)
        else:
            print("NOT ELI5, use exact match to preprocess")
            data = preprocess_data(filepath)
        output_file = os.path.join(output_dir, os.path.basename(filepath))
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            print("Dump {} data into {}".format(split_name, output_file))

    