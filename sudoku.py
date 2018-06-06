#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:07:45 2018

@author: melizalab
"""

def seqorder(n=10,t=10):
    import numpy as np
    init = np.array([1])
    if n%2 == 0:
        for i in range(n//2):
            init = np.append(init,[i+2,n-i])
    else:
        for i in range((n-1)//2):
            init = np.append(init,[i+2,n-i])
        init = np.append(init,(n-1)//2)
    square = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                square[i,j] = init[j]
            else:
                if (square[i-1,j])%n != 0:
                    square[i,j] = square[i-1,j]+1
                else:
                    square[i,j] = 1
    if n%2 == 1:
        square = np.append(square,np.fliplr(square),axis=0)
    if len(square) < t:
        square = np.append(square,np.fliplr(square),axis=0)
        square = square[:t,:]
    elif len(square) > t:
        square = square[:t,:]
    order = np.matrix.flatten(square)
    order = order.astype(int)
    return order

            
