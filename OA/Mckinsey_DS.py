#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 07:56:58 2017

@author: Yiming
"""

def  jobOffers(scores, lowerLimits, upperLimits):
    ranges = []
    for i in range(len(lowerLimits)):
        ranges.append([lowerLimits[i], upperLimits[i]])
        
    result = []
    for interval in ranges:
        result.append(sum([(score >= interval[0])& (score <= interval[1])for score in scores]))

    return result




def gradient_descent(alpha,x1, x2, x3, y, ep = 0.01, max_iter = 10000):
    converged  = False
    iter = 0
    t0 = 1 * len(x1) 
    t1 = 1 * len(x1) 
    t2 = 1 * len(x1) 
    t3 = 1 * len(x1) 
    
    J = sum([(t0 + t1 * x1[i] + t2 * x2[i] + t3 * x3[i] - y[i])**2 for i in range(len(y))]) #Initial Error
    
    while not converged:
        
        grad0 = 1.0/len(y) * (sum([(t0 + t1 * x1[i] + t2 * x2[i] + t3 * x3[i] - y[i]) for i in range(len(y))]))
        grad1 = 1.0/len(y) * (sum([(t0 + t1 * x1[i] + t2 * x2[i] + t3 * x3[i] - y[i]) * x1[i] for i in range(len(y))]))
        grad2 = 1.0/len(y) * (sum([(t0 + t1 * x1[i] + t2 * x2[i] + t3 * x3[i] - y[i]) * x2[i] for i in range(len(y))]))
        grad3 = 1.0/len(y) * (sum([(t0 + t1 * x1[i] + t2 * x2[i] + t3 * x3[i] - y[i]) * x3[i] for i in range(len(y))]))
        
        t0 = t0 - alpha * grad0
        t1 = t1 - alpha * grad1
        t2 = t2 - alpha * grad2
        t3 = t3 - alpha * grad3
        
        e = sum([(t0 + t1 * x1[i] + t2 * x2[i] + t3 * x3[i] - y[i]) **2 for i in range(len(y))])
        
        if abs(J-e)<0.0001:
            print("Converged successfully")
            converged = True
        
        
        if iter==max_iter:
            converged = True
        
        J=e
        iter+=1
        
    return t0, t1, t2, t3

def predictMissingHumidity(startDate, endDate, knownTimestamps, humidity, timestamps):
    start_y = int(startDate.split("-")[0])
    start_m = int(startDate.split("-")[1])
    start_d = int(startDate.split("-")[2])
    end_y = int(endDate.split("-")[0])
    end_m = int(endDate.split("-")[1])
    end_d = int(endDate.split("-")[2])
    
    known_m = []
    known_d = []
    known_h = []
    
    for known in knownTimestamps:
        known_m.append(int(known.split(" ")[0].split("-")[1])) # month
        known_d.append(int(known.split(" ")[0].split("-")[2])) # day
        known_h.append(int(known.split(" ")[1].split(":")[0])) # hour
        
    unknown_m = []
    unknown_d = []
    unknown_h = []
    
    for unknown in timestamps:
        unknown_m.append(int(unknown.split(" ")[0].split("-")[1])) # month
        unknown_d.append(int(unknown.split(" ")[0].split("-")[2])) # day
        unknown_h.append(int(unknown.split(" ")[1].split(":")[0])) # hour
        
    t0 ,t1, t2, t3 = gradient_descent(0.2, known_m, known_d, known_h, humidity, ep = 0.001, max_iter = 1000)
    
    pred = []
    for i in range(len(unknown_m)):
        pred.append(t0 + t1 * unknown_m[i] + t2 * unknown_d[i] + t3 * unknown_h[i])
        
    return pred



a = [[1,2],[3,4],[5,6]]
sum1 = [0] * 3
for a1 in a[-3:]:
    for j in range(3):
        sum1[j] += a1[j]
    
    





def predictTemperature(startDate, endDate, temperature, n):
    
 
