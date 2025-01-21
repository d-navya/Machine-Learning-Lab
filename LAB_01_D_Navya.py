#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Write a program to count the number of vowels and consonants present in input string
def NumberOfVowAndConsInString(string):
    list1= []
    for s in string:
        list1.append(s)
    # initializing vowels and consonants as 0 and 0
    vowels, consonants= 0, 0
    vow= ['a', 'i', 'e', 'o', 'u']
    # checking if vowel is present in string
    for s in list1:
        if s in vow:
            vowels+= 1
        else:
            consonants+= 1
    result= [vowels, consonants]
    return result


# In[2]:


#Write a program that accepts two matrices A and B as input and returns their product AB, Check if A and B are multipliable; if not return error message
def MatrixMul(A, B):
    row1= len(A)
    column1= len(A[0])
    row2= len(B)
    column2= len(B[0])
    # checking for condition valid or not
    if column1!= row2:
        return "invalid input"
    result = [[0] * column2 for _ in range(row1)]
    # Perform matrix multiplication
    for i in range(row1):
        for j in range(column2):
            for k in range(column1):
                result[i][j] += A[i][k] * B[k][j]
    return result


# In[70]:


#Write a program to find the number of common elements between two lists. The lists contain integers
def CommonElements(list1, list2):
    dict1= {}
    dict2= {}
    # storing frequency of occurence of first list
    for num1 in list1:
        if num1 in dict1:
            dict1[num1]+= 1
        else:
            dict1[num1]= 1
    print(dict1)
    # storing frequency of occurence of second list
    for num2 in list2:
        if num2 in dict2:
            dict2[num2]+= 1
        else:
            dict2[num2]= 1
    print(dict2)
    count= 0
    # adding the frequency of the common list occurences
    for num in dict1.keys():
        if num in dict2:
            count+= min(dict1[num], dict2[num])
    return count


# In[77]:


#Write a rogram that accepts a matrix as input and return its transpose
def Transpose(A):
    x= len(A)
    y= len(A[0])
    b= [[0]*x for _ in range(y)]
    # exchanging the elements for transpose
    for i in range(x):
        for j in range(y):
            b[i][j]= A[j][i]
    return b


# In[78]:


NumberOfVowAndConsInString("send it to me rachel send it to me please")


# In[79]:


MatrixMul([[1, 2, 3], [0, 2, 1], [1, 2, 5]], [[1, 0], [0, 1], [2, 1]])


# In[80]:


CommonElements([2, 2, 4, 5, 6], [3, 2, 2, 5, 8, 10])


# In[81]:


Transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# In[ ]:




