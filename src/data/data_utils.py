import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir), os.path.pardir)))
from tasks.dense_retriever.tokenizers import SimpleTokenizer
from spacy.lang.en import English

nlp = None

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
    have_answers = []
    for i, sent in enumerate(doc.sents):
        sent = str(sent)
        sentences.append(sent)
        starts, ends, _ = mark_begin_end_span_in_text(answer, sent)
        have_answers.append(len(starts))
    new_text = ' '.join(sentences)
    starts = []
    ends = []
    current_sentence_start = 0
    last_have_answer = False
    for sent, answer_number in zip(sentences, have_answers):
        if answer_number > 0:
            if not last_have_answer:
                starts.append(current_sentence_start)
                ends.append(current_sentence_start + len(sent) - 1)
                last_have_answer = True
            else:
                ends[-1] += (1 + len(sent))
        else:
            last_have_answer = False
        current_sentence_start += len(sent) + 1 # 1 is the length of space
    return starts, ends, new_text
         


def mark_begin_end_span_in_text(answer, text):
    simple_tokenizer = SimpleTokenizer()
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

def count_sentence_length(dataset):
    sentence_length = []
    for d in dataset:
        extractive_info = d['extractive']
        for index, have_answer in enumerate(extractive_info['local_have_answer']):
            if have_answer == 0:
                continue
            else:
                local_starts = extractive_info['local_context_starts'][index]
                local_ends = extractive_info['local_context_ends'][index]
                for s, e in zip(local_starts, local_ends):
                    sentence_length.append(e - s + 1)
    sentence_length.sort()
    print("mean length:", sum(sentence_length) / len(sentence_length))
    print("min length:", min(sentence_length))
    for i in range(1, 10):
        print("{}0% sentence length:".format(i), sentence_length[int(len(sentence_length) * i / 10)])
    print("max length:", max(sentence_length))


if __name__ == "__main__":
    # debug
    # text = "2002 Gordon Wharmby Gordon Wharmby (6 November 1933 – 18 May 2002) was a British television actor. He 2002 was best known for the role  of Wesley Pegden on \"Last of the Summer Wine\".    He was born in Manchester, Lancashire, in 1933, and served in the Royal Air Force during his national service.Wharmby was 1.223 originally employed as a painter and decorator and had no formal training as an actor. He gained stage experience with Oldham Repertory Theatre and worked part-time as a jobbing actor.\tEarly television roles included bit-parts in programmes such as \"Bill Brand\" (1976) 2002, \"The One and Only Phyllis"
    text = 'forgive [himself] if [he] made others as miserable as [he] made [himself]." Lanser argues that the same argument of devastation and misery can be said about the work of Edgar Allan Poe, but his work is still printed and studied by academics. "The Yellow Wallpaper" provided feminists the tools to interpret literature in different ways. Lanser argues that the short story was a "particularly congenial medium for such a re-vision . . . because the narrator herself engages in a form of feminist interpretation when she tries to read the paper on her wall". The narrator in the story is'
    title = "Gordon Wharmby"
    question = "when did wesley leave last of the summer wine"
    answer = "2002"
    starts, ends, text = mark_begin_end_sentence_in_text(answer, text)
    print(starts, ends)
    print(len(text))
    for s, e in zip(starts, ends):
        print(text[s:e + 1])
        print('"' + text[e] + '"')
