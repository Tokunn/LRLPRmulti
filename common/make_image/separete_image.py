#!/usr/bin/env python2

import os, shutil
import random
random.seed(1)

IMGSDIR = 'imgs'
TRAINDIR = 'train'
TESTDIR = 'test'

def preparedirs():
    if (os.path.exists(TRAINDIR)):
        shutil.rmtree(TRAINDIR)
    os.mkdir(TRAINDIR)
    if (os.path.exists(TESTDIR)):
        shutil.rmtree(TESTDIR)
    os.mkdir(TESTDIR)

def chkmkdir(dirpath):
    if (not os.path.exists(dirpath)):
        os.mkdir(dirpath)
        print("make dir : " + dirpath)

def main():
    preparedirs()

    numdirs = os.listdir(IMGSDIR)
    for num in numdirs:
        numpath = os.path.join(IMGSDIR, num)
        imgnamelist = os.listdir(numpath)
        random.shuffle(imgnamelist)
        i = 0

        for img in imgnamelist:
            srcpath = os.path.join(numpath, img)

            if (i % 5 == 0):
                testnumpath = os.path.join(TESTDIR, num)
                dstpath = os.path.join(testnumpath, img)
                chkmkdir(testnumpath)
            else:
                trainnumpath = os.path.join(TRAINDIR, num)
                dstpath = os.path.join(os.path.join(TRAINDIR, num), img)
                chkmkdir(trainnumpath)

            print("MOVE " + srcpath + " to " + dstpath)
            shutil.move(srcpath, dstpath)
            i+=1

if __name__ == '__main__':
    main()
