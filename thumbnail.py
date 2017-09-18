# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:55:07 2017

@author: Chris
"""
import os, sys
from PIL import Image
size = 32, 32

for infile in sys.argv[1:]:
    outfile = os.path.splitext(infile)[0] + ".thumbnail"
    if infile != outfile:
        try:
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(outfile, "JPEG")
        except IOError:
            print ("cannot create thumbnail for '%s'" % infile)
            
            
            