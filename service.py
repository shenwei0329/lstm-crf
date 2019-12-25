# coding: utf-8
#

import argparse
import os
import time
import numpy as np
import tensorflow as tf
from data import read_corpus, read_dictionary, tag2label, random_embedding
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity, get_entity2
from helper import transfer_corpus
from data import data_list
from restapi import ltp_service

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

## hyperparameters
## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random',
                            help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo/text')
# parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
parser.add_argument('--demo_model', type=str, default='1550144205', help='model for test and demo')
parser.add_argument('--text_file', type=str, default='my.txt', help='text file for demo')
args = parser.parse_args()

## get char embeddings
word2id = read_dictionary('./data_path/word2id.pkl')
embeddings = random_embedding(word2id, 300)
output_path = './data_path_save/1577156952'
model_path = os.path.join(output_path, "checkpoints/")
ckpt_prefix = os.path.join(model_path, "model")
ckpt_file = tf.train.latest_checkpoint(model_path)

## paths setting
paths = {}
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path

model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
model.build_graph()
saver = tf.compat.v1.train.Saver()

sess = tf.Session(config=config)
saver.restore(sess, ckpt_file)

## text
def do_lstm(sentence):

    global sess, model

    per = []
    loc = []
    org = []

    # with tf.Session(config=config) as sess:
    if sess is not None:
        print('============= demo =============')

        if len(sentence)>10:

            demo_sent = sentence.replace(u"　",u"，").replace(u"《", "").replace(u"》", "").replace(" ", "").replace(",", u"，").replace(u"［", "")
            demo_sent = demo_sent.replace(u"］",u"").replace(u"（", "").replace(u"）", "").replace(u"—", "").replace(u"〔"," ").replace(u"〕", " ")
            demo_sent = demo_sent.replace(u"＂",u"").replace(u"“", "").replace(u"”", "").replace("...", "").replace(u"⒄", "")

            _sent = [demo_sent]
            for _s in _sent:
                if len(_s) < 10:
                    continue
                # print("{}".format(_s))
                _sent = list(_s.strip())
                _data = [(_sent, ['O'] * len(_sent))]
                # print(_data)
                tag = model.demo_one(sess, _data)
                try:
                    PER, LOC, ORG = get_entity(tag, _sent)
                    # print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
                    if len(PER)>0:
                        for _p in PER:
                            if len(_p)>1 and _p not in per:
                                print('PER: {}'.format(_p))
                                per.append(_p)
                    if len(LOC)>0:
                        for _p in LOC:
                            if len(_p)>1 and _p not in loc:
                                print('LOC: {}'.format(_p))
                                loc.append(_p)
                    if len(ORG)>0:
                        for _p in ORG:
                            if len(_p)>1 and _p not in org:
                                print('ORG: {}'.format(_p))
                                org.append(_p)
                except Exception as e:
                    print(e)

    return per, loc, org

