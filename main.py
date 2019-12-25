import argparse
import os
import time
# import pandas as pd
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
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')

## read corpus and get training data
if args.mode != 'demo' and args.mode != 'text':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)

## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data + "_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

## training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)


## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.compat.v1.train.Saver()
    comment_data = pd.read_csv('data/first50_comment.csv', encoding='utf-8')
    # new_comments = comment_data.apply(transfer_corpus, axis=1)
    new_comments = comment_data['评论'].tolist()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        # result = {'position':[], 'symptom':[]}
        # for item in new_comments:
        #     demo_sent = list(item.strip())
        #     demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        #     tag = model.demo_one(sess, demo_data)
        #     POS, SYM = get_entity(tag, demo_sent)
        #     print(POS, SYM)
        #     result['position'].append(POS)
        #     result['symptom'].append(SYM)
        # print(result)
        # while (1):
        #     print('Please input your sentence:')
        #     demo_sent = input()
        #     if demo_sent == '' or demo_sent.isspace():
        #         print('See you next time!')
        #         break
        #     else:
        #         demo_sent = list(demo_sent.strip())
        #         demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        #         tag = model.demo_one(sess, demo_data)
        #         PER, LOC, ORG = get_entity(tag, demo_sent)
        #         print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
        result = []
        for data_sent in new_comments[:5]:
            ret = {}
            demo_sent = list(data_sent.strip())
            demo_data = [(demo_sent, ['O'] * len(demo_sent))]
            tag = model.demo_one(sess, demo_data)
            POS, SYM = get_entity2(tag, demo_sent)
            # ret['sent'] = data_sent
            # ret['position'] = POS
            # ret['symptom'] = SYM
            # result.append(ret)
            if len(POS)>0:
                print(data_sent)
                print('POS:{}\nSYM:{}'.format(POS, SYM))
        # df = pd.DataFrame(data=result)
        # df.to_csv('data/result.csv', encoding='utf_8_sig')

## text
elif args.mode == 'text':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.compat.v1.train.Saver()

    per = []
    loc = []
    org = []

    f = open(args.text_file, 'r', encoding='utf-8')
    new_comments = f.readlines()
    f.close()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)

        for demo_sent in new_comments:
            demo_sent = demo_sent.replace(u"　",u"，").replace(u"《", "").replace(u"》", "").replace(" ", "").replace(",", u"，").replace(u"［", "")
            demo_sent = demo_sent.replace(u"］",u"").replace(u"（", "").replace(u"）", "").replace(u"—", "").replace(u"〔"," ").replace(u"〕", " ")
            demo_sent = demo_sent.replace(u"＂",u"").replace(u"“", "").replace(u"”", "").replace("...", "").replace(u"⒄", "")

            if len(demo_sent) < 10:
                continue
            # _sent = ltp_service(demo_sent)
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

    f = open("out.txt", "w", encoding='utf-8')
    f.write("PER:\n")
    for _p in sorted(per):
        f.write(_p)
        f.write("\n")

    f.write("LOC:\n")
    for _p in sorted(loc):
        f.write(_p)
        f.write("\n")

    f.write("ORG:\n")
    for _p in sorted(org):
        f.write(_p)
        f.write("\n")

    f.close()


