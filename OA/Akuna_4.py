#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:23:26 2017

@author: Yiming
"""

def number_base_arithmetic(str1, str2):
    base = int(str1)
    if base < 2 or base > 36:
        print("number invalid")
    
    elif base <= 10:
        if all([int(s) in list(range(base)) for s in str2]):
            return int(str2, base)
        else: 
            print("number invalid")
        
         
    elif base >= 11 or base <= 36:
        myset1 = [str(x) for x in list(range(10))]
        myset2 = [chr(y) for y in list(range(97, base - 10 + 97))]
        
        if all([s in myset1 + myset2 for s in str2]):
            return int(str2, base)
        else:
            print("number invalid")
                

number_base_arithmetic("1", "1010")
number_base_arithmetic("8", "19")
number_base_arithmetic("2", "1010")
number_base_arithmetic("11", "a1")
number_base_arithmetic("12", "c33")


def find_palindromes(year):
    century_start = int(str(year)[0:2]) * 100 + 1
    count = 0
    
    for i in list(range(century_start, century_start + 100)):
        current_year = str(i)
        digit_7 = current_year[::-1] + current_year[1:4]
        digit_8 = current_year[::-1] + current_year
        
        if ((int(digit_7[0]) in [1, 3, 5, 7, 8, 10, 12]) and (int(digit_7[1:3]) in list(range(1, 32)))) or ((int(digit_7[0]) in [4, 6, 9, 11]) and (int(digit_7[1:3]) in list(range(1, 31)))) or ((int(digit_7[0]) in [2]) and (int(digit_7[1:3]) in list(range(1, 29)))):
            count = count + 1
            print(digit_7)
        
        if ((digit_8[0:2] in [str(s).zfill(2) for s in [1, 3, 5, 7, 8, 10, 12]]) and (digit_8[2:4] in [str(s).zfill(2) for s in list(range(1, 32))])) or ((digit_8[0:2] in [str(s).zfill(2) for s in [2, 4, 6, 9, 11]]) and (digit_8[2:4] in [str(s).zfill(2) for s in list(range(1, 31))])) or ((digit_8[0:2] in [str(s).zfill(2) for s in [2]]) and (digit_8[2:4] in [str(s).zfill(2) for s in list(range(1, 29))])):
            count = count + 1
            print(digit_8)
            
    return count
    
 
def word_graph(words, tuples):
    result = []
    tuple_list = []
    for i in range(97, 123):
        tuple_list.append(chr(i) + chr(i))
        
    for sub_tuple in tuples:
        tuple_list.append(sub_tuple[0] + sub_tuple[1])
        
    for word in words:
        if len(word) == 1:
            result.append(1)
            
        elif all([x in tuple_list for x in [word[i: i+2] for i in range(len(word) - 1)]]):
            result.append(1)
            
        else:
            result.append(0)
        
    return result
    
    
    
    
    
    
    
    
