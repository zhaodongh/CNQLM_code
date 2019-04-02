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
dataset = FLAGS.data
isEnglish = FLAGS.isEnglish


def cut(sentence, isEnglish=isEnglish):
    if isEnglish:
        tokens =sentence.lower().split()
        tokens = [word for word in sentence.split() if word not in stopwords]
    else:
        tokens = [word for word in sentence.split() if word not in stopwords]
    return tokens


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

def prepare(cropuses, is_embedding_needed=False, dim=50, fresh=False):
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
                    embeddings_1 = KeyedVectors.load_word2vec_format(fname, binary=True)
                    sub_embeddings = getSubVectors(embeddings_1,alphabet,dim)
                    embedding_complex = getSubVectors_complex_random(alphabet,dim)
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
        return alphabet, sub_embeddings,embedding_complex
    else:
        return alphabet
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
def load(dataset = dataset, filter = False):
    data_dir = "../data/" + dataset
    datas = []  
    for data_name in ['train.txt','dev.txt','test.txt']:
        if data_name=='train.txt':
            data_file = os.path.join(data_dir,data_name)
            data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna("WASHINGTON")
            if filter == True:
                datas.append(removeUnanswerdQuestion(data))
            else:
                datas.append(data)
        if data_name=='dev.txt':
            data_file = os.path.join(data_dir,data_name)
            data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna("WASHINGTON")
            if filter == True:
                datas.append(removeUnanswerdQuestion(data))
            else:
                datas.append(data)
        if data_name=='test.txt':
            data_file = os.path.join(data_dir,data_name)
            data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna("WASHINGTON")
            if filter == True:
                datas.append(removeUnanswerdQuestion(data))
            else:
                datas.append(data)
    
    sub_file = os.path.join(data_dir,'submit.txt')
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
        q_position = position_index(row['question'], q_len)
        a_position = position_index(row['answer'], a_len)
        overlap=overlap_score(row['question'],row['answer'])
        pairs.append((quetion, answer, q_position, a_position,overlap,q_overlap,a_overlap))
    n_batches = int(len(pairs) * 1.0 / batch_size)
    for i in range(0, n_batches):
        batch = pairs[i * batch_size:(i + 1) * batch_size]

        yield [[pair[j] for pair in batch] for j in range(input_num)]
    batch = pairs[n_batches * batch_size:] + [pairs[n_batches *
                                                    batch_size]] * (batch_size - len(pairs) + n_batches * batch_size)
    yield [[pair[i] for pair in batch] for i in range(input_num)]
def batch_gen_with_point_wise(df, alphabet, batch_size=10, overlap_dict=None, q_len=33, a_len=40):
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
        pairs.append((question, answer, label,q_position, a_position,overlap,q_overlap,a_overlap))
    n_batches = int(len(pairs) * 1.0 / batch_size)
    pairs = sklearn.utils.shuffle(pairs, random_state=121)

    for i in range(0, n_batches):
        batch = pairs[i * batch_size:(i + 1) * batch_size]
        yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]
    batch = pairs[n_batches * batch_size:] + [pairs[n_batches *
                                                    batch_size]] * (batch_size - len(pairs) + n_batches * batch_size)
    yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]
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
