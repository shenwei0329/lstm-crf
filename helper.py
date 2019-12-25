# coding: utf-8
#

import sys
import pickle
import re


def write2word2id(comment=None):
    """
    得到语料中文的词向量坐标
    ：return dict 返回值为字典
    """
    # comment = comment.upper()
    words = set(''.join(comment))

    # 用户也可以在此进行自定义过滤字符
    # r1 = u'[a-zA-Z0-9’!"#　$	%&\'()*+,-./:;<=>?@，。·α?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    r1 = u'[’!"#　$	%&\'()*+,-./:;<=>?@，。·α?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

    words = re.sub(r1, '', ''.join(words))
    print(u"words: {}".format(words))
    word2id = {}
    for num, word in enumerate(words):
        word2id[word] = num
    word2id['<UNK>'] = 0
    word2id = sorted(word2id.items(), key=lambda d: d[1])
    from collections import OrderedDict
    word2id = OrderedDict(word2id)
    with open('t_word2id.pkl', mode='wb') as f:
        pickle.dump(word2id, f)  


if __name__ == '__main__':

    _str = u""
    with open(sys.argv[1], "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) < 1:
                break
            line = line.replace("\n", '').replace("\r", '')
            line = str(line).split('\t')[0]
            _str = _str + line
        f.close()

    write2word2id(_str)

    """
    write2word2id(u"我们当前的Hadoop产品能力主要是解决数据从分区规划、数据汇聚、数据治理到服务交付的全流程的管理上。"
                  u"当前我们的Spark产品主要用于将行业客户的异构多源的结构化数据汇聚到大数据平台体系上进行治理和加工，"
                  u"并以数据服务的形式支撑应用")
    """
