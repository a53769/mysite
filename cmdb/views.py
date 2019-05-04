from django.shortcuts import render
from django.shortcuts import HttpResponse
# Create your views here.
from cmdb.algorithm_breast.Cluster import getImage
from cmdb.models import Img,Txt
from cmdb.algorithm_breast.Caption import generate
import pandas as pd
import os
import cmdb.algorithm_breast.TxtProcess as TP

# def index(request):
#     # return HttpResponse("Hello World!")
#     if request.method == "POST":
#         # username = request.POST.get("username", None)
#         # password = request.POST.get("password", None)
#         # temp = {"user": username, "pwd": password}
#         # user_list.append(temp)
#         file_obj = request.FILES.get('img', None)
#
#         if file_obj == None:
#             return HttpResponse('file not existing in the request')
#         img = Img(img_url=request.FILES.get('img'), name=request.FILES.get('img').name)
#         img.save()
#         # print(img.img_url.path)
#         images = getImage(img.img_url.path, img.img_url.url)
#
#         content = {
#             # "data":user_list,
#             "imgs": images
#         }
#         return render(request, "start.html", content)
#
#     return render(request, 'start.html', )


def home(request):
    return render(request, 'home.html', )


def textSeg(request):
    if request.method == "POST":

        file_obj = request.FILES.get('txt', None)

        if file_obj == None:
            return HttpResponse('file not existing in the request')

        filename = request.FILES.get('txt').name
        seqs_file = Txt(txt_url=request.FILES.get('txt'), name=filename)
        seqs_file.save()
        seg_file = seqs_file.txt_url.path

        # 获取原始文本序列以及分词后的序列
        seqs_org, seqs_seg = TP.get_TextSeg(seg_file)

        # 将分词后的单词拼接成序列，以“|”隔开
        for index, sentence in enumerate(seqs_seg):
            seqs_seg[index] = '|'.join(sentence)

        data = zip(seqs_org, seqs_seg )
        content = {
            # "data":user_list,
            "data": data,
        }
        return render(request, 'textSeg.html', content)
    return render(request, 'textSeg.html', )


def textTag(request):
    if request.method == "POST":

        file_obj = request.FILES.get('txt', None)

        if file_obj == None:
            return HttpResponse('file not existing in the request')

        filename = request.FILES.get('txt').name
        seqs_file = Txt(txt_url=request.FILES.get('txt'), name=filename)
        seqs_file.save()

        org_file = seqs_file.txt_url.path
        seg_file = r"./media/resource_breastTxt/predict_seg.txt"

        # 获取原始文本序列以及分词后的序列
        seqs_org, seqs_seg = TP.get_TextSeg(org_file)

        # 将分词后的单词拼接成序列，以“ ”隔开, 并存入指定路径文件seg_file
        TP.save_file(seg_file, seqs_seg)

        # 获取标注后的序列, (待预测文件即指定路径文件seg_file，在config.py文件中修改)
        seqs_tag = TP.get_TextTag()

        # 将 seg 和 tag 序列合并为一个字符串
        seqs_seg_tag = []
        for seqs_index in range(len(seqs_tag)):
            seqs_comb = []
            for tag_index,tag in enumerate(seqs_tag[seqs_index]):
                word = seqs_seg[seqs_index][tag_index]
                seqs_comb.append(word + '/' + tag)
            seqs_seg_tag.append(seqs_comb)

        data = zip(seqs_org, seqs_seg, seqs_tag, seqs_seg_tag)

        content = {
            # "data":user_list,
            "data": data,
        }
        return render(request, 'textTag.html', content)
    return render(request, 'textTag.html', )


def textStruct(request):
    return render(request, 'textStruct.html', )


def img2Text(request):
    if request.method == "POST":
        file_obj = request.FILES.get('img', None)
        if file_obj == None:
            return HttpResponse('file not existing in the request')
        img = Img(img_url=request.FILES.get('img'), name=request.FILES.get('img').name)
        img.save()

        img_url = img.img_url.url
        img_path = img.img_url.path

        text, original_text, classify, label = generate(img_path,img.img_url.name)

        content = {
            # "data":user_list,
            "img": img_url,
            "original_text": original_text,
            "text": text,
            "classify": 'BI-Rads:'+ str(classify),
            "labels": label
        }
        return render(request, 'img2Text.html', content)
    return render(request, 'img2Text.html', )


def imgClusterSeg(request):
    if request.method == "POST":
        # username = request.POST.get("username", None)
        # password = request.POST.get("password", None)
        # temp = {"user": username, "pwd": password}
        # user_list.append(temp)
        file_obj = request.FILES.get('img', None)

        if file_obj == None:
            return HttpResponse('file not existing in the request')
        img = Img(img_url=request.FILES.get('img'), name=request.FILES.get('img').name)
        img.save()
        # print(img.img_url.path)
        images = getImage(img.img_url.path, img.img_url.url)

        content = {
            # "data":user_list,
            "imgs": images
        }
        return render(request, "imgClusterSeg.html", content)

    return render(request, 'imgClusterSeg.html', )


def others(request):
    return render(request, 'others.html', )
