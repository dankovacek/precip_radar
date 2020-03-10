import os
import sys
from PIL import Image

from os import listdir
from os.path import isfile, join


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PNG_DIR = os.path.join(BASE_DIR, 'png_radar_images/')


for subdir, dirs, files in os.walk(BASE_DIR):
    for d in dirs:
        if '17' in d:
            print(d)
            mypath = BASE_DIR + '/' + d
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            print(onlyfiles)
            print("")
            for f in onlyfiles:
                im = Image.open(mypath + '/' + f)
                im.save(PNG_DIR + f + '.png')
            # for subdir, dirs, files in os.walk(BASE_DIR + '/' + d)
            # im = Image.open(d + '/' + ):
            # print(d + '/' + file))
#         f=open(file,'r')
#         lines=f.readlines()
#         f.close()
#         f=open(file,'w')
#         for line in lines:
#             newline = "No you are not"
#             f.write(newline)
#         f.close()
# im = Image.open('Foto.jpg')
# im.save('Foto.png')
