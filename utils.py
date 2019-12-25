import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    ITEM = get_ITEM_entity(tag_seq, char_seq)
    DT = get_DT_entity(tag_seq, char_seq)
    # POS = get_POS_entity(tag_seq, char_seq)
    # SYM = get_SYM_entity(tag_seq, char_seq)
    return PER, LOC, ORG
    # return POS, SYM


def get_entity2(tag_seq, char_seq):
    # PER = get_PER_entity(tag_seq, char_seq)
    # LOC = get_LOC_entity(tag_seq, char_seq)
    # ORG = get_ORG_entity(tag_seq, char_seq)
    POS = get_POS_entity(tag_seq, char_seq)
    SYM = get_SYM_entity(tag_seq, char_seq)
    # return PER, LOC, ORG
    return POS, SYM


def get_POS_entity(tag_seq, char_seq):
    length = len(char_seq)
    POS = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-P':
            if 'pos' in locals().keys():
                POS.append(pos)
                del pos
            pos = char
            if i + 1 == length:
                POS.append(pos)
        if tag == 'I-P':
            try:
                pos += char
            except UnboundLocalError:
                pos = char
            if i + 1 == length:
                POS.append(pos)
        if tag not in ['I-P', 'B-P']:
            if 'pos' in locals().keys():
                POS.append(pos)
                del pos
            continue
    return POS


def get_SYM_entity(tag_seq, char_seq):
    length = len(char_seq)
    SYM = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-S':
            if 'sym' in locals().keys():
                SYM.append(sym)
                del sym
            sym = char
            if i+1 == length:
                SYM.append(sym)
        if tag == 'I-S':
            try:
                sym += char
            except UnboundLocalError:
                sym = char
            if i+1 == length:
                SYM.append(sym)
        if tag not in ['I-S', 'B-S']:
            if 'sym' in locals().keys():
                SYM.append(sym)
                del sym
            continue
    return SYM


def get_ITEM_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ITEM':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-ITEM':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-ITEM', 'B-ITEM']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_DT_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-DT':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-DT':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-DT', 'B-DT']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
