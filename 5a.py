#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:30:02 2018

@author: liuxiaoqin
"""
import matplotlib.pyplot as plt
import numpy as np

fl = open("5adata.txt","r")
xdata =[]
ydata =[]
count = 1000
for i in fl.readlines():
    [float(s) for s in i.split() if s.isdigit()]
    s = float(s)
    xdata.append(count)
    ydata.append(s)
    count += 1000
    
fig1 = plt.figure(20)
plt.plot(xdata, ydata)
plt.ylabel("average return")
plt.xlabel("episodes")
plt.title("Part5a: training curve")
fl.close()