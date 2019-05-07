# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import numpy as np

model_path = "./media/models/adModels/clinical_m.pkl"
CN = np.array([[1,0,0]])
MCI = np.array([[0,1,0]])
AD = np.array([[0,0,1]])

def model_predict(result):

    clinical_data = [list(result.values())]
    clf = joblib.load(model_path)  # 模型地址
    y = clf.predict(clinical_data)  # [[0,0,0,0,0,0,0,0,0,0,0]]传入指标

    if ((y == CN).all()):
        result = "正常（CN）"
    if ((y == MCI).all()):
        result = "轻度认知障碍（MCI）"
    else:# ((y == AD).all())
        result = "阿尔茨海默病（AD）"
    return result
#
# if __name__ == "__main__":
#     demo_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#     demo_path = "/x/x/x/x/x"
#     print(model_predict(demo_path, demo_data))
