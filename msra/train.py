# coding=utf-8
#! /usr/bin/env python3.4
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from helper_my import get_overlap_dict, batch_gen_with_point_wise, load, prepare, batch_gen_with_single,test_my
import operator
from QA_CNN_point import QA_quantum
import random
import evaluation
import pickle
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
now = int(time.time())
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

from functools import wraps

FLAGS = config.flags.FLAGS
FLAGS._parse_flags()
#FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
log_dir = 'wiki_log/' + timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
para_file = log_dir + '/test_' + FLAGS.data + timeStamp + '_para'
precision = data_file + 'precise'

pickle.dump(FLAGS.__flags, open(para_file, 'wb+'))

def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco
#@log_time_delta
def predict(sess, cnn, test, alphabet, batch_size, q_len, a_len):
    scores = []
    d = get_overlap_dict(test, alphabet, q_len, a_len)
    for data in batch_gen_with_single(test, alphabet, batch_size, q_len, a_len, overlap_dict=d):
        feed_dict = {
            cnn.question: data[0],
            cnn.answer: data[1],
            cnn.q_position:data[2],
            cnn.a_position:data[3],
            cnn.overlap:data[4],
            cnn.q_overlap:data[5],
            cnn.a_overlap:data[6]
        }
        score = sess.run(cnn.scores, feed_dict)
        scores.extend(score)
    return np.array(scores[:len(test)])

@log_time_delta
def test_point_wise():
    train, dev, test = load(FLAGS.data, filter=FLAGS.clean)  #wiki
    # train, test, dev = load(FLAGS.data, filter=FLAGS.clean) #trec
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print(q_max_sent_length)
    print(a_max_sent_length)
    print(len(train))
    print ('train question unique:{}'.format(len(train['question'].unique())))
    print ('train length', len(train))
    print ('test length', len(test))
    print ('dev length', len(dev))

    alphabet, embeddings,embeddings_complex = prepare(
        [train, test, dev], max_sent_length=a_max_sent_length,dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
    print(embeddings_complex)
    print ('alphabet:', len(alphabet))
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            # session_conf = tf.ConfigProto(
            #     allow_soft_placement=FLAGS.allow_soft_placement,
            #     log_device_placement=FLAGS.log_device_placement)

            session_conf = tf.ConfigProto()
            session_conf.allow_soft_placement = FLAGS.allow_soft_placement
            session_conf.log_device_placement = FLAGS.log_device_placement
            session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default(), open(precision, "w") as log:
            log.write(str(FLAGS.__flags) + '\n')
            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            cnn = QA_quantum(
                max_input_left=q_max_sent_length,
                max_input_right=a_max_sent_length,
                vocab_size=len(alphabet),
                embedding_size=FLAGS.embedding_dim,
                batch_size=FLAGS.batch_size,
                embeddings=embeddings,
                embeddings_complex=embeddings_complex,
                dropout_keep_prob=FLAGS.dropout_keep_prob,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                is_Embedding_Needed=True,
                trainable=FLAGS.trainable,
                overlap_needed=FLAGS.overlap_needed,
                position_needed=FLAGS.position_needed,
                pooling=FLAGS.pooling,
                hidden_num=FLAGS.hidden_num,
                extend_feature_dim=FLAGS.extend_feature_dim)
            cnn.build_graph()
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            starter_learning_rate = FLAGS.learning_rate
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step, 100, 0.96)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            # optimizer =  tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            sess.run(tf.global_variables_initializer())
            map_max = 0.65
            now = int(time.time())
            timeArray = time.localtime(now)
            timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
            timeDay = time.strftime("%Y%m%d", timeArray)
            print (timeStamp)
            for i in range(FLAGS.num_epochs):
                d = get_overlap_dict(
                    train, alphabet, q_len=q_max_sent_length, a_len=a_max_sent_length)
                datas = batch_gen_with_point_wise(train, alphabet, FLAGS.batch_size, overlap_dict=d,
                                                  q_len=q_max_sent_length, a_len=a_max_sent_length)
                for data in datas:
                    feed_dict = {
                        cnn.question: data[0],
                        cnn.answer: data[1],
                        cnn.input_y: data[2],
                        cnn.q_position:data[3],
                        cnn.a_position:data[4],
                        cnn.overlap:data[5],
                        cnn.q_overlap:data[6],
                        cnn.a_overlap:data[7]
                    }
                    _, step, loss, accuracy, pred, scores,input_y,position= sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy,cnn.predictions, cnn.scores,cnn.input_y,cnn.embedding_W_pos],feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}  ".format( time_str, step, loss, accuracy))
                now = int(time.time())
                timeArray = time.localtime(now)
                timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
                timeDay = time.strftime("%Y%m%d", timeArray)
                print (timeStamp)
                # print(position_embedding)
                # np.savetxt("position_embedding_xiangwei_uniform.txt",position_embedding)
                predicted = predict(
                    sess, cnn, train, alphabet, FLAGS.batch_size, q_max_sent_length, a_max_sent_length)
                predicted_label=np.argmax(predicted,1)
                map_mrr_train = evaluation.evaluationBypandas_f1_acc(
                    train, predicted[:, -1],predicted_label)
                predicted_test = predict(
                    sess, cnn, test, alphabet, FLAGS.batch_size, q_max_sent_length, a_max_sent_length)
                predicted_label=np.argmax(predicted_test,1)
                map_mrr_test = evaluation.evaluationBypandas_f1_acc(
                    test, predicted_test[:, -1],predicted_label)
                if map_mrr_test[0] > map_max:
                    map_max = map_mrr_test[0]
                    timeStamp = time.strftime(
                        "%Y%m%d%H%M%S", time.localtime(int(time.time())))
                    folder = 'runs/' + timeDay
                    out_dir = folder + '/' + timeStamp + \
                        '__' + FLAGS.data + str(map_mrr_test[0])
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    #save_path = saver.save(sess, out_dir)
                print ("{}:train epoch:map mrr {}".format(i, map_mrr_train))
                print ("{}:test epoch:map mrr {}".format(i, map_mrr_test))
                line2 = " {}:epoch: map_test{}".format(i, map_mrr_test)
                log.write(line2+'\n')
                log.flush()
            log.close()


if __name__ == '__main__':
    # test_quora()
    if FLAGS.loss == 'point_wise':
        test_point_wise()
    # test_pair_wise()
    # test_point_wise()
