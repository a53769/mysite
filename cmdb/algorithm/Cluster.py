import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from numpy.core.multiarray import ndarray

from sklearn.cluster import KMeans
import numpy as np

from skimage import morphology
from skimage import filters, segmentation, measure


IN_C = 1024
IN_R = 1024


def get_mask(Image):
    thresh = filters.threshold_otsu(Image)  # 阈值分割
    bw = morphology.closing(Image > thresh, morphology.disk(1))
    # plt.imshow(bw, cmap=plt.cm.gray)
    # plt.show()
    print('mask')
    return bw


def get_ROI(pre: ndarray, image: ndarray) -> ndarray:
    roi_image = np.zeros((IN_R, IN_C))
    for i in range(IN_R):
        for j in range(IN_C):
            if pre[i, j]:
                roi_image[i, j] = image[i, j]
    # plt.imshow(roi_image, cmap=plt.cm.gray)
    # plt.show()
    print('roi')
    return roi_image


def get_feature(roi_image):
    pad_image = np.pad(roi_image, ((1, 1), (1, 1)), 'constant')
    feature1 = []
    feature2 = []
    feature3 = []
    for i in range(1, IN_R + 1):
        for j in range(1, IN_C + 1):
            x1 = pad_image[i - 1, j - 1]
            x2 = pad_image[i - 1, j]
            x3 = pad_image[i - 1, j + 1]
            x4 = pad_image[i, j - 1]
            x5 = pad_image[i, j]
            x6 = pad_image[i, j + 1]
            x7 = pad_image[i + 1, j - 1]
            x8 = pad_image[i + 1, j]
            x9 = pad_image[i + 1, j + 1]
            feature1.append((x2 + x4 + x6 + x8) / 4)
            feature2.append((x2 + x4 + x6 + x8 + x1 + x3 + x7 + x9) / 8)
            feature3.append((x2 + x4 + x6 + x8 + x1 + x3 + x5 + x7 + x9) / 9)
    data__feature1 = np.reshape(np.array(feature1), (IN_R * IN_C, 1))
    data__feature2 = np.reshape(np.array(feature2), (IN_R * IN_C, 1))
    data__feature3 = np.reshape(np.array(feature3), (IN_R * IN_C, 1))
    DataVec = np.reshape(roi_image, (IN_R * IN_C, 1))
    print('features')
    return np.c_[np.c_[DataVec, data__feature1], np.c_[data__feature2, data__feature3]]


def get_cluster_area(FeatureVec):
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(FeatureVec)  # 进行聚类
    pre = kmeans.predict(FeatureVec)
    pre.resize(1024, 1024)
    pre[pre == 0] = 7
    # plt.imshow(pre, cmap=plt.cm.gray)
    print('cluster')
    return pre


def get_max_intensity(cluster_area, roi_image):

    bw = morphology.opening(cluster_area, morphology.disk(1))  # 闭运算 使用边长为3的正方形进行形态滤波填充孔洞

    cleared = bw.copy()  # 复制
    segmentation.clear_border(cleared)  # 清除与边界相连的目标物

    label_image = measure.label(cleared)  # 连通区域标记
    borders = np.logical_xor(bw, cleared)  # 异或
    label_image[borders] = -1

    regions = measure.regionprops(label_image, intensity_image=roi_image, coordinates='rc')
    max_regions = [regions[0]]

    for region in regions:  # 循环得到每一个连通区域属性集
        # 忽略小区域
        if region.area < 120 or region.area > 90000:
            continue
        # 离心率>0.3
        if region.eccentricity < 0.3:
            continue
        # 占空比>0.26
        if region.extent < 0.29:
            continue
        # 可靠性>0.5
        if region.solidity < 0.71:
            continue
        # max_regions.append(region)
        if max_regions[0].max_intensity < region.max_intensity:
            if len(max_regions):
                del max_regions[:]
            max_regions = [region]
        elif max_regions[0].max_intensity == region.max_intensity:
            max_regions.append(region)
    print('regions')
    return max_regions


def get_mass(max_regions, image, name):
    if len(max_regions):
        for index, region in enumerate(max_regions):
            minr, minc, maxr, maxc = region.bbox
            plt.imshow(image[minr:maxr, minc:maxc], cmap='gray')
            plt.savefig(name)
        print('save')
    else:
        print("NO MASS")


def get_rect(max_regions, aug, param):
    if len(max_regions):
        for index, region in enumerate(max_regions):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            fig.tight_layout()
            plt.imshow(aug)

            plt.savefig(param)
        print('save')
    else:
        print("NO MASS")


def AugImge(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_array = clahe.apply(img)

    return img_array


def getImage(path, url):
    imgs = []
    img = cv2.imread(path,0)
    cv2.imwrite(path[0:-3]+'jpg', img)
    imgs.append(url[0:-3]+'jpg')
    aug = AugImge(img)
    cv2.imwrite(path[0:-4] + '_aug.jpg', aug)
    imgs.append(url[0:-4]+'_aug.jpg')
    max_regions = get_max_intensity(get_cluster_area(get_feature(get_ROI(get_mask(aug), aug))), aug)
    get_mass(max_regions, aug, path[0:-4] + '_mass.jpg')
    imgs.append(url[0:-4] + '_mass.jpg')
    get_rect(max_regions, aug, path[0:-4] + '_rect.jpg')
    imgs.append(url[0:-4] + '_rect.jpg')

    return imgs