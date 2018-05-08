#!/usr/bin/env python2
# coding: utf-8

# In[ ]:


import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageFilter
import PIL.ImageOps
import matplotlib.pyplot as plt
import os, random, time, datetime, math, sys, shutil
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img


# Initialize imgs folder
imgdir = './imgs/'
if (os.path.exists(imgdir)):
    shutil.rmtree(imgdir)
os.mkdir(imgdir)

# Set image count
argvs = sys.argv
argc = len(argvs)
if (argc != 2):
    number_of_plates = 100
else:
    number_of_plates = int(argvs[1])
print("Make " + str(number_of_plates) + " images")

# Set font path
color = (8, 57, 62)
fontpath1 = "/usr/share/fonts/TTF/TakaoPGothic.ttf"
fontpath2 = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
if (os.path.exists(fontpath1)):
    fontpath = fontpath1
elif (os.path.exists(fontpath2)): 
    fontpath = fontpath2
else:
    print("FONT not found in " + fontpath1 + " and " + fontpath2)

#if not (os.path.exists(fontpath)):
#    print("Font not found in " + fontpath)
#    fontpath = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
#    if not (os.path.exists(fontpath)):
#        print("Font not found in " + fontpath)


def draw_number(img, text):
    draw = PIL.ImageDraw.Draw(img)
    draw.font = PIL.ImageFont.truetype(fontpath, 115, encoding='unic')
    
    img_size = np.array(img.size)
    txt_size = np.array(draw.font.getsize(text))
    pos = (img_size - txt_size) / 2
    draw.text(np.array([50, 55]), text, fill=color)
    
def draw_char(img, text):
    draw = PIL.ImageDraw.Draw(img)
    draw.font = PIL.ImageFont.truetype(fontpath, 40, encoding='unic')
    
    img_size = np.array(img.size)
    txt_size = np.array(draw.font.getsize(text))
    draw.text(np.array([15, 105]), text, fill=color)
    
   
def draw_country(img, text, num):
    draw = PIL.ImageDraw.Draw(img)
    draw.font = PIL.ImageFont.truetype(fontpath, 40, encoding='unic')
    
    img_size = np.array(img.size)
    text += ' ' + num
    txt_size = np.array(draw.font.getsize(text))
    pos = (img_size - txt_size) / 2
    draw.text(np.array([pos[1], 10]), text, fill=color)
 


## In[ ]:
#
#
#img = PIL.Image.new("RGBA", (400, 200))
#num = "12-44"
#char = u"あ"
#country = u"宇都宮"
#country_num = u"230"
#draw_number(img, num)
#draw_char(img, char)
#draw_country(img, country, country_num)
#
#fil_par = 20
#
#plt.imshow(np.asarray(img))
#plt.show()
#
#filtered_img = img.filter(PIL.ImageFilter.GaussianBlur(radius=fil_par))
#plt.imshow(np.asarray(filtered_img))
#plt.show()
#
#filtered_img = img.resize((img.width/fil_par, img.height/fil_par))
#plt.imshow(np.asarray(filtered_img))
#plt.show()


# In[ ]:


def make_license_plates(country, country_number, char, number):
    img = PIL.Image.new("RGB", (400, 200), color=(255,255,255))
    draw_country(img, country, country_number)
    draw_char(img, char)
    draw_number(img, number)
    #plt.imshow(np.asarray(img))
    #plt.show()
    
    if not(os.path.exists(imgdir + '{0}'.format(number[:2]+number[3:]))):
        os.mkdir(imgdir + '{0}'.format(number[:2]+number[3:]))
    for i in range(1, 15, 4):
        #img_filed = img.filter(PIL.ImageFilter.GaussianBlur(radius=i))
        img_filed = img.resize((img.width/i, img.height/i))
        img_filed.save(imgdir + '{0}/{1}.png'.format(number[:2]+number[3:], i))
        #plt.imshow(np.asarray(img_filed))
        #plt.show()
        
        # extend
        train_datagen = ImageDataGenerator(rotation_range=5,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.1,
                                  #zoom_range=[1,.5],
                                  fill_mode='nearest')
        train_generator = train_datagen.flow(np.asarray([img_to_array(img_filed)]),
                                             batch_size=1,
                                             save_to_dir=imgdir + '{0}'.format(number[:2]+number[3:]),
                                             save_prefix='extend_{0}'.format(i), save_format='png')
        for j in range(30):
        #for j in range(2):
            #plt.imshow(array_to_img(train_generator.next()[0]))
            #plt.show()
            train_generator.next()


# In[ ]:


random.seed(1)

country_temp = [u"福島", u"会津", u"郡山", u"いわき", u"水戸", u"土浦",
                u"つくば", u"宇都宮", u"那須", u"とちぎ", u"群馬", u"前橋", u"高崎"] * int(math.ceil(number_of_plates/10.0))

country_list = random.sample(country_temp, number_of_plates)
#country_list = country_temp[:number_of_plates]

country_number_list = [random.randrange(111, 999) for i in range(number_of_plates)]
#country_number_list = list(range(111, 999, 13)[:number_of_plates])

char_temp = u"あ い う え お か き く け こ さ し す せ そ た ち つ て と な に ぬ ね の は ひ ふ へ こ ま み む め も や ゆ よ わ を ん".split() * int(math.ceil(number_of_plates/41.0))
char_list = random.sample(char_temp, number_of_plates)
#char_list = char_temp[:number_of_plates]

#number_list = [random.randrange(1111, 9999) for i in range(number_of_plates)]
#number_list = list(range(1111, 9999, 234)[:number_of_plates])
number_list = ['9052', '2223', '9282', '5140', '3802', '5366', '5495', '1683', '2694', '3277', '4082', '1327', '4581', '3837', '9851', '9864', '4019', '9137', '3289', '3229', '8778', '5922', '3606', '3719', '3413', '9203', '6091', '6407', '3300', '4675', '8176', '2340', '2514', '1796', '9173', '5810', '4814', '6906', '8294', '7611', '8583', '8677', '8478', '7046', '6010', '4121', '1753', '9472', '1778', '8009', '4646', '3355', '3186', '7212', '7464', '6257', '5562', '1393', '1741', '9054', '9547', '4775', '4628', '1664', '6877', '5682', '3403', '2370', '6856', '7743', '8151', '8255', '9673', '3695', '8640', '3111', '3953', '6830', '2060', '6554', '5006', '1834', '6182', '3920', '2650', '3842', '5511', '6758', '4265', '1177', '5948', '2481', '5567', '7240', '4956', '3926', '5091', '5992', '1292', '8412']
number_list = random.sample(number_list, number_of_plates)


# In[ ]:


#print(datetime.datetime.now())
start_time = time.time()

for i in range(0, number_of_plates):
    make_license_plates(country_list[i],
                        str(country_number_list[i]),
                        char_list[i],
                        str(number_list[i])[:2]+'-'+str(number_list[i])[2:])
    
#print(datetime.datetime.now())
print(time.time() - start_time)
print("number_of_plates : {0}".format(number_of_plates))
