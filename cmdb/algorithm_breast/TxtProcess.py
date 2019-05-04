import jieba
from cmdb.algorithm.rnn import textTaggingMain

def get_TextSeg(path, pred_file):
    org = []
    seg = []
    seg_save = []
    jieba.load_userdict(r"./media/usr_dics.txt")  # 加载自定义词典
    with open(path, 'r', encoding="utf-8") as file:
        for sentences in file.readlines():
            org.append(sentences)
            seg_words = jieba.cut(sentences)  # 切分成词默认是精确模式
            seg_res = " ".join(seg_words)
            string_seg = "|".join(seg_res.split())
            seg_save.append(seg_res)
            seg.append(string_seg)

    with open(pred_file, 'w', encoding="utf-8") as file:
        for item in seg_save:
            file.writelines(item)
    return org,seg

def get_TextTag(path):
    data = textTaggingMain(path)
    return data