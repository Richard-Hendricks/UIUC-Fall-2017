#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:23:26 2017

@author: Yiming
"""

# Complete the function below.

def number_base_arithmetic(base, number):
    mybase = int(base)
    # check base
    if mybase < 2 or mybase > 36: 
        return "base invalid"
    
    # check number part 1 (all numbers)
    elif mybase <= 10:
        if all([int(s) in list(range(mybase)) for s in number]):
            return str(int(number, mybase))
        else: 
            return "number invalid"
        
    # check number part 2 (numbers & letters)
    elif mybase >= 11 or mybase <= 36:
        myset1 = [str(x) for x in list(range(10))] # number set
        myset2 = [chr(y) for y in list(range(97, mybase - 10 + 97))] # chr(97) = "a"
        
        if all([s in myset1 + myset2 for s in number]):
            return str(int(number, mybase))
        else:
            return "number invalid"

                

number_base_arithmetic("1", "1010")
number_base_arithmetic("8", "19")
number_base_arithmetic("2", "1010")
number_base_arithmetic("11", "a1")
number_base_arithmetic("12", "c33")


# Complete the function below.

def  find_palindromes(year):
    century_start = int(str(year)[0:2]) * 100 + 1
    count = 0
    
    for i in list(range(century_start, century_start + 100)):
        current_year = str(i)
        digit_7 = current_year[::-1] + current_year[1:4]
        digit_8 = current_year[::-1] + current_year
        
        if ((int(digit_7[0]) in [1, 3, 5, 7, 8, 10, 12]) and (int(digit_7[1:3]) in list(range(1, 32)))) or ((int(digit_7[0]) in [4, 6, 9, 11]) and (int(digit_7[1:3]) in list(range(1, 31)))) or ((int(digit_7[0]) in [2]) and (int(digit_7[1:3]) in list(range(1, 29)))):
            count = count + 1
        
        if ((digit_8[0:2] in [str(s).zfill(2) for s in [1, 3, 5, 7, 8, 10, 12]]) and (digit_8[2:4] in [str(s).zfill(2) for s in list(range(1, 32))])) or ((digit_8[0:2] in [str(s).zfill(2) for s in [2, 4, 6, 9, 11]]) and (digit_8[2:4] in [str(s).zfill(2) for s in list(range(1, 31))])) or ((digit_8[0:2] in [str(s).zfill(2) for s in [2]]) and (digit_8[2:4] in [str(s).zfill(2) for s in list(range(1, 29))])):
            count = count + 1
            
    return count

find_palindromes(2016)   
    

def  find_words_in_graph(list_of_words, edge_list):
    result = []
    tuple_list = [] # store all edges
    # self-self (duplicate letters: "aa", "pp")
    for i in range(97, 123):
        tuple_list.append(chr(i) + chr(i))
    # store as "ab" 
    for edge in edge_list:
        tuple_list.append(edge[0] + edge[1])
    # length = 1: always succeed
    for word in list_of_words:
        if len(word) == 1:
            result.append(1)
        # check other edges
        elif all([x in tuple_list for x in [word[i: i+2] for i in range(len(word) - 1)]]):
            result.append(1)
            
        else:
            result.append(0)
        
    return result
            
            
               
    
    
