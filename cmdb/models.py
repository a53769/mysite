from django.db import models

# Create your models here.


class Img(models.Model):
    img_url = models.ImageField(upload_to='img') # upload_to指定图片上传的途径，如果不存在则自动创建
    name = models.CharField(max_length=20)


class Txt(models.Model):
    txt_url = models.FileField(upload_to='txt') # upload_to指定图片上传的途径，如果不存在则自动创建
    # txt_url = models.FilePathField(path='txt') # path指定图片上传的途径，如果不存在则自动创建
    name = models.CharField(max_length=20)