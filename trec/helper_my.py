# -*- coding:utf-8-*-
import numpy as np
import random,os,math
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
import sklearn
import multiprocessing
import time
import pickle 
from collections import defaultdict
import evaluation
import string
from nltk import stem
from tqdm import tqdm
import chardet
import re
import config
from functools import wraps
import nltk
from nltk.corpus import stopwords
from numpy.random import seed
import math
seed(1234)
FLAGS = config.flags.FLAGS
FLAGS._parse_flags()
#FLAGS.flag_values_dict()
dataset = FLAGS.data
isEnglish = FLAGS.isEnglish
UNKNOWN_WORD_IDX = 0
is_stemmed_needed = False
# stopwords=stopwords.words("english")
# stopwords = { word.decode("utf-8") for word in open("model/chStopWordsSimple.txt").read().split()}


def cut(sentence, isEnglish=isEnglish):
    if isEnglish:
        tokens =sentence.lower().split()
        # tokens = [word for word in sentence.split() if word not in stopwords]
    else:
        # words = jieba.cut(str(sentence))
        tokens = [word for word in sentence.split() if word not in stopwords]
    return tokens
#print( tf.__version__)


def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%s runed %.2f seconds" % (func.__name__, delta))
        return ret
    return _deco


class Alphabet(dict):
    def __init__(self, start_feature_id=1):
        self.fid = start_feature_id

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
      # self[idx] = item
            self.fid += 1
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))


@log_time_delta
def prepare(cropuses, max_sent_length=31,is_embedding_needed=False, dim=50, fresh=False):
    vocab_file = 'model/voc'

    if os.path.exists(vocab_file) and not fresh:
        alphabet = pickle.load(open(vocab_file, 'r'))
    else:
        alphabet = Alphabet(start_feature_id=0)
        alphabet.add('[UNKNOW]')
        alphabet.add('END')
        count = 0
        for corpus in cropuses:
            for texts in [corpus["question"].unique(), corpus["answer"]]:
                for sentence in tqdm(texts):
                    count += 1
                    if count % 10000 == 0:
                        print (count)
                    tokens = cut(sentence)
                    for token in set(tokens):
                        alphabet.add(token)
        print (len(alphabet.keys()))
        alphabet.dump('alphabet_clean.txt')
        #pickle.dump(alphabet,open(vocab_file,'wb+'))
    if is_embedding_needed:
        sub_vec_file = '../embedding/sub_vector'
        if os.path.exists(sub_vec_file) and not fresh:
            sub_embeddings = pickle.load(open(sub_vec_file, 'r'))
        else:
            if isEnglish:
                if dim == 50:
                    fname = "../embedding/aquaint+wiki.txt.gz.ndim=50.bin"
                    #fname = "../embedding/glove.6B.100d.txt"
                    embeddings_1 = KeyedVectors.load_word2vec_format(fname, binary=True)
                    sub_embeddings = getSubVectors(embeddings_1,alphabet,dim)
                    embedding_complex = getSubVectors_complex_uniform(max_sent_length,dim)
                else:
                    fname = "../embedding/glove.6B.100d.txt"
                    embeddings_1= load_text_vec(alphabet, fname, embedding_size=dim)
                    sub_embeddings = getSubVectorsFromDict(embeddings_1,alphabet,dim)
                    embedding_complex = getSubVectors_complex_random(alphabet,dim)
            else:
                fname = 'model/wiki.ch.text.vector'
                embeddings = load_text_vec(alphabet, fname, embedding_size=dim)
                sub_embeddings = getSubVectorsFromDict(
                    embeddings, alphabet, dim)
            pickle.dump(sub_embeddings, open(sub_vec_file, 'wb'))
        # print (len(alphabet.keys()))
        # embeddings = load_vectors(vectors,alphabet.keys(),layer1_size)
        # embeddings = KeyedVectors.load_word2vec_format(fname, binary=True)
        # sub_embeddings = getSubVectors(embeddings,alphabet)
        #print(sub_embeddings[49240],embedding_complex[49240])
        #exit()
        return alphabet, sub_embeddings,embedding_complex
    else:
        return alphabet

def load_text_vec_complex(alphabet,filename="",datafile='',embedding_size = 100):
    vectors = {}
    embedding_alphabet=[]
    file1=pd.read_csv(filename,sep='\t',names=["word","id"])
    file2=np.load(datafile)
    for i in range(len(file1)):
        word = file1['word'][i]
        if word in alphabet:
            #vectors={'vocab':word,'vec':file["number"][i]}
            vectors[word] = file2[i].astype(np.float)
            embedding_alphabet.append(word)
    return vectors,embedding_alphabet
  
def getSubVectors_complex(vectors,vocab,embedding_alphabet,dim = 100):
    #file = open('missword','wb',encoding='utf-8')'"+list[0]+"
    #embedding = np.zeros((len(vocab), 100))
    temp_vec = 0
    embeddings=[]
    for word in vocab:
        if word in embedding_alphabet:
            #print(vectors[eval('word')])
            #exit()
            #print(map(float,vectors[eval('word')]))
            #print(type(str2float(map(float,vectors[eval('word')]))))
            #exit()
            #embedding[i]= vectors[eval('word')]
            embeddings.append(vectors[eval('word')])
        else:
            #embedding[i]= np.random.uniform(-0.25,+0.25,100)
            embeddings.append(np.random.uniform(-0.25,+0.25,100))
    return embeddings

def get_lookup_table(embedding_params):
    id2word = embedding_params['id2word']
    word_vec = embedding_params['word_vec']
    lookup_table = []

    # Index 0 corresponds to nothing
    lookup_table.append([0]* embedding_params['wvec_dim'])
    for i in range(1, len(id2word)):
        word = id2word[i]
        wvec = [0]* embedding_params['wvec_dim']
        if word in word_vec:
            wvec = word_vec[word]
        # print(wvec)
        lookup_table.append(wvec)

    lookup_table = np.asarray(lookup_table)
    return(lookup_table)
def getSubVectors(vectors, vocab, word_embe,dim=50):
    embedding = np.zeros((len(vocab), dim))
    temp_vec = 0
    for word in vocab:
        if word in vectors.vocab:
            embedding[vocab[word]] = vectors.word_vec(word)
        else:
            # .tolist()
            embedding[vocab[word]
                      ] = np.random.uniform(-0.25,+0.25,vectors.syn0.shape[1])
        temp_vec += embedding[vocab[word]]
    temp_vec /= len(vocab)
    for index, _ in enumerate(embedding):
        embedding[index] -= temp_vec
    return embedding
def transform(flag):
    if flag == 1:
        return [0, 1]
    else:
        return [1, 0]
def getSubVectors_complex_random(vocab, dim=50):
    embedding = np.zeros((len(vocab), dim))
    for word in vocab:#RandomUniform(minval=0, maxval=2*math.pi)
        embedding[vocab[word]] = np.random.uniform(0, +2*math.pi, dim)
    return embedding


def getSubVectors_complex_uniform(max_sentence, dim=50):
    embedding = np.zeros((max_sentence, dim))
    for i in range(max_sentence):
        embedding[i] = np.random.uniform(+((2*math.pi)/30)*i, +((2*math.pi)/30)*(i+1), dim)
    return embedding
'''def getSubVectors(vectors, vocab, dim=50):
    print ('embedding_size:', vectors.syn0.shape[1])
    embedding = np.zeros((len(vocab), vectors.syn0.shape[1]))
    temp_vec = 0
    for word in vocab:
        if word in vectors.vocab:
            embedding[vocab[word]] = vectors.word_vec(word)
        else:
            # .tolist()
            embedding[vocab[word]
                      ] = np.random.uniform(-0.25, +0.25, vectors.syn0.shape[1])
        embedding_ave=np.sum(embedding[vocab[word]])/dim
        embedding[vocab[word]]=embedding[vocab[word]]-embedding_ave
        temp_vec += embedding[vocab[word]]
    temp_vec /= len(vocab)
    for index, _ in enumerate(embedding):
        embedding[index] -= temp_vec
    return embedding'''
def load_text_vec(alphabet, filename="", embedding_size=100):
    vectors = {}
    with open(filename) as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print ('epch %d' % i)
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size = items[0], items[1]
                print (vocab_size, embedding_size)
            else:
                word = items[0]
                if word in alphabet:
                    vectors[word] = items[1:]
    print ('embedding_size', embedding_size)
    print ('done')
    print ('words found in wor2vec embedding ', len(vectors.keys()))
    return vectors


def getSubVectorsFromDict(vectors, vocab, dim=300):
    file = open('missword', 'w')
    embedding = np.zeros((len(vocab), dim))
    count = 1
    for word in vocab:

        if word in vectors:
            count += 1
            embedding[vocab[word]] = vectors[word]
        else:
            # if word in names:
            #     embedding[vocab[word]] = vectors['谁']
            # else:
            file.write(word + '\n')
            # vectors['[UNKNOW]'] #.tolist()
            embedding[vocab[word]] = np.random.uniform(-0.5, +0.5, dim)
    file.close()
    print ('word in embedding', count)
    return embedding


@log_time_delta
def get_overlap_dict(df, alphabet, q_len=40, a_len=40):
    d = dict()
    for question in df['question'].unique():
        group = df[df['question'] == question]
        answers = group['answer']
        for ans in answers:
            q_overlap, a_overlap = overlap_index(question, ans, q_len, a_len)
            d[(question, ans)] = (q_overlap, a_overlap)
    return d
# calculate the overlap_index


def overlap_index(question, answer, q_len, a_len, stopwords=[]):
    qset = set(cut(question))
    aset = set(cut(answer))

    q_index = np.zeros(q_len)
    a_index = np.zeros(a_len)

    overlap = qset.intersection(aset)
    for i, q in enumerate(cut(question)[:q_len]):
        value = 1
        if q in overlap:
            value = 2
        q_index[i] = value
    for i, a in enumerate(cut(answer)[:a_len]):
        value = 1
        if a in overlap:
            value = 2
        a_index[i] = value
    return q_index, a_index


def position_index(sentence, length):
    index = np.zeros(length)
    raw_len = len(cut(sentence))
    index[:min(raw_len, length)] = range(1, min(raw_len + 1, length + 1))
    # print index
    return index


def encode_to_split(sentence, alphabet, max_sentence=40):
    indices = []
    tokens = cut(sentence)
    for word in tokens:
        indices.append(alphabet[word])
    while(len(indices) < max_sentence):
        indices += indices[:(max_sentence - len(indices))]
    # results=indices+[alphabet["END"]]*(max_sentence-len(indices))
    return indices[:max_sentence]

def overlap_score(question,answer):
    question_word=cut(question)
    answer_word=cut(answer)
    same_num=0
    for w in question_word:
        if w in answer_word:
            same_num = same_num+1
        else:
            same_num += 0
    return [same_num,1]



# def load(dataset = dataset, filter = False):
#     data_dir = "../data/" + dataset
#     datas = []  
#     for data_name in ['train.txt','dev.txt','test.txt']:
#         if data_name=='train.txt':
#             data_file = os.path.join(data_dir,data_name)
#             data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna("WASHINGTON")
#             if filter == True:
#                 datas.append(removeUnanswerdQuestion(data))
#             else:
#                 datas.append(data)
#         if data_name=='dev.txt':
#             data_file = os.path.join(data_dir,data_name)
#             data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna("WASHINGTON")
#             if filter == True:
#                 datas.append(removeUnanswerdQuestion(data))
#             else:
#                 datas.append(data)
#         if data_name=='test.txt':
#             data_file = os.path.join(data_dir,data_name)
#             data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna("WASHINGTON")
#             if filter == True:
#                 datas.append(removeUnanswerdQuestion(data))
#             else:
#                 datas.append(data)
    
#     sub_file = os.path.join(data_dir,'submit.txt')
#     return tuple(datas)

def load(dataset=dataset, filter=False):
    data_dir = "../data/" + dataset
    datas = []
    for data_name in ['train-all.txt', 'test.txt']:
        data_file = os.path.join(data_dir, data_name)
        data = pd.read_csv(data_file, header=None, sep="\t", names=[
                           "qid", "aid", "question", "answer", "flag"], quoting=3).fillna('N')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    data_file = os.path.join(data_dir, "dev.txt")
    dev = pd.read_csv(data_file, header=None, sep="\t", names=[
                      "question", "answer", "flag"], quoting=3).fillna('N')
    datas.append(dev)
    # submit = pd.read_csv(sub_file,header = None,sep = "\t",names = ['question','answer'],quoting = 3)
    # datas.append(submit)
    return tuple(datas)


def removeUnanswerdQuestion(df):
    counter = df.groupby("question").apply(lambda group: sum(group["flag"]))
    questions_have_correct = counter[counter > 0].index
    counter = df.groupby("question").apply(
        lambda group: sum(group["flag"] == 0))
    questions_have_uncorrect = counter[counter > 0].index
    counter = df.groupby("question").apply(lambda group: len(group["flag"]))
    questions_multi = counter[counter > 1].index

    return df[df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()


@log_time_delta
def batch_gen_with_single(df, alphabet, batch_size=10, q_len=33, a_len=40, overlap_dict=None):
    pairs = []
    input_num = 7
    for index, row in df.iterrows():
        quetion = encode_to_split(
            row["question"], alphabet, max_sentence=q_len)
        answer = encode_to_split(row["answer"], alphabet, max_sentence=a_len)
        if overlap_dict == None:
            q_overlap, a_overlap = overlap_index(
                row["question"], row["answer"], q_len, a_len)
        else:
            q_overlap, a_overlap = overlap_dict[(
                row["question"], row["answer"])]
        # overlap_scores=overlap_score(row["question"],row["answer"])
        # q_position = index*(q_len+a_len)+position_index(row['question'], q_len)
        # a_position = index*(q_len+a_len)+position_index(row['answer'], a_len)        
        q_position = position_index(row['question'], q_len)
        a_position = position_index(row['answer'], a_len)
        overlap=overlap_score(row['question'],row['answer'])
        pairs.append((quetion, answer, q_position, a_position,overlap,q_overlap,a_overlap))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches = int(len(pairs) * 1.0 / batch_size)
    # pairs = sklearn.utils.shuffle(pairs,random_state =132)
    for i in range(0, n_batches):
        batch = pairs[i * batch_size:(i + 1) * batch_size]

        yield [[pair[j] for pair in batch] for j in range(input_num)]
    batch = pairs[n_batches * batch_size:] + [pairs[n_batches *
                                                    batch_size]] * (batch_size - len(pairs) + n_batches * batch_size)
    yield [[pair[i] for pair in batch] for i in range(input_num)]
def batch_gen_with_point_wise(df, alphabet, batch_size=10, overlap_dict=None, q_len=33, a_len=40):
    # inputq inputa intput_y overlap
    input_num = 8
    pairs = []
    for index, row in df.iterrows():
        question = encode_to_split(
            row["question"], alphabet, max_sentence=q_len)
        answer = encode_to_split(row["answer"], alphabet, max_sentence=a_len)
        if overlap_dict == None:
            q_overlap, a_overlap = overlap_index(
                row["question"], row["answer"], q_len, a_len)
        else:
            q_overlap, a_overlap = overlap_dict[(
                row["question"], row["answer"])]
        q_position = position_index(row['question'], q_len)
        a_position = position_index(row['answer'], a_len)
        label = transform(row["flag"])
        overlap=overlap_score(row['question'],row['answer'])
        #pairs.append((question, answer, label, q_overlap,a_overlap, q_position, a_position))
        pairs.append((question, answer, label,q_position, a_position,overlap,q_overlap,a_overlap))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches = int(len(pairs) * 1.0 / batch_size)
    pairs = sklearn.utils.shuffle(pairs, random_state=121)

    for i in range(0, n_batches):
        batch = pairs[i * batch_size:(i + 1) * batch_size]
        yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]
    batch = pairs[n_batches * batch_size:] + [pairs[n_batches *
                                                    batch_size]] * (batch_size - len(pairs) + n_batches * batch_size)
    yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]


# def batch_gen_with_point_wise(df, alphabet, batch_size=10, overlap_dict=None, q_len=33, a_len=40):
#     # inputq inputa intput_y overlap
#     input_num = 6
#     pairs = []
#     for question,group in df.groupby("question"):
#         i=0
#         pos_answer=group[group["flag"]==1]["answer"]
#         neg_answer=group[group["flag"]==0]["answer"]
#         if len(pos_answer)==0 or len(neg_answer)==0:
#             continue
#         for pos in pos_answer:
#             i=0
#             neg_index=np.random.choice(neg_answer.index)
#             neg=neg_answer.loc[neg_index,]
#             label_neg=transform(0)
#             seq_neg_a=encode_to_split(neg,alphabet,max_sentence=a_len)
#             seq_pos_a=encode_to_split(pos,alphabet,max_sentence=a_len)
#             label_pos=transform(1)
#             question_seq=encode_to_split(question,alphabet,max_sentence=q_len)
#             q_position_neg = position_index(question, q_len)
#             q_position_pos = position_index(question, q_len)
#             a_position_neg = position_index(neg, a_len)
#             a_position_pos = position_index(pos, a_len)
#             overlap_pos=overlap_score(question,pos)
#             overlap_neg=overlap_score(question,neg)
#             pairs.append((question_seq,seq_neg_a,label_neg,q_position_neg,a_position_neg,overlap_neg))
#             pairs.append((question_seq,seq_pos_a,label_pos,q_position_pos,a_position_pos,overlap_pos))
#             i=i+1
#     n_batches = int(len(pairs) * 1.0 / batch_size)
#     pairs = sklearn.utils.shuffle(pairs, random_state=121)

#     for i in range(0, n_batches):
#         batch = pairs[i * batch_size:(i + 1) * batch_size]
#         yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]
#     batch = pairs[n_batches * batch_size:] + [pairs[n_batches *
#                                                     batch_size]] * (batch_size - len(pairs) + n_batches * batch_size)
#     yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]
def test_my(sentence,alphabet,q_len):
    question=encode_to_split(sentence,alphabet,max_sentence=q_len)

    return question

if __name__ == '__main__':
    train, test, dev = load(FLAGS.data, filter=FLAGS.clean)
    q_max_sent_length = max(
        map(lambda x: len(x), train['question'].str.split()))
    a_max_sent_length = max(map(lambda x: len(x), train['answer'].str.split()))
    alphabet, embeddings = prepare(
        [train, test, dev], dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
