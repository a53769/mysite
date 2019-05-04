import jieba
import os
from cmdb.algorithm.rnn import textTag
from cmdb.algorithm.config import *


def save_file(path_file, seqs_seg_list):
    # 将分词后的单词拼接成序列，以“ ”隔开, 并存入指定路径文件
    with open(path_file, 'w', encoding="utf-8") as file:
        for words in seqs_seg_list:
            seq_words = ' '.join(words)
            file.writelines(seq_words)


def get_TextSeg(seg_file):
    org_list = []
    seg_list = []
    jieba.load_userdict(os.path.join(resourcePath, "usr_dics.txt"))  # 加载自定义词典
    with open(seg_file, 'r', encoding="utf-8") as file:
        for sentences in file.readlines():
            org_list.append(sentences)
            words = [item for item in jieba.cut(sentences)]  # 切分成词默认是精确模式
            seg_list.append(words)
    return org_list, seg_list

def get_TextTag():
    return textTag()