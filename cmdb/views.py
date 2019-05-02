from django.shortcuts import render
from django.shortcuts import HttpResponse
# Create your views here.
from cmdb.algorithm.Cluster import getImage
from cmdb.models import Img
from cmdb.algorithm.Caption import generate
import pandas as pd


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


def textTagging(request):
    return render(request, 'textTagging.html', )


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
