#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:11:12 2020

@author: lewis
"""

import naco_pip
from naco_pip import input_dataset

test = input_dataset('/home/lpic0002/pd87_scratch/products/NACO_archive/5_HD179218/raw/',
                     '/home/lpic0002/pd87_scratch/products/NACO_archive/5_HD179218/calib/')

test.bad_columns()
test.mk_dico()
test.find_sky_in_sci_cubes()
