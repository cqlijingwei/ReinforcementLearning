#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 21:54:10 2018

@author: liuxiaoqin
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

fl = open("5b1.txt","r")
fl2 = open("5b2.txt","r")
fl3 = open("5b3.txt","r")
xdata =[]
ydata =[]
count = 1000
for i in fl.readlines():
    [float(s) for s in i.split() if s.isdigit()]
    s = float(s)
    xdata.append(count)
    ydata.append(s)
    count+=1000
    
xdata1 =[]
ydata1 =[]
count = 1000
for i in fl2.readlines():
    [float(s1) for s1 in i.split() if s1.isdigit()]
    s1 = float(s1)
    xdata1.append(count)
    ydata1.append(s1)
    count+=1000

xdata2 =[]
ydata2 =[]
count = 1000
for i in fl3.readlines():
    [float(s2) for s2 in i.split() if s2.isdigit()]
    s2 = float(s2)
    xdata2.append(count)
    ydata2.append(s2)
    count+=1000
    
fig1 = plt.figure(20)
patch1 = mpatches.Patch(color='orange', label='hidden size 16')
patch2 = mpatches.Patch(color='blue', label='hidden size 32')
patch3 = mpatches.Patch(color='green', label='hidden size 128')
plt.legend(handles=[patch1,patch2,patch3])
plt.plot(xdata, ydata, label="hidden size 32")
plt.plot(xdata1, ydata1, label="hidden size 16")
plt.plot(xdata2, ydata2, label="hidden size 128")
plt.ylabel("average return")
plt.xlabel("episodes")
plt.title("Part5b: training curve")