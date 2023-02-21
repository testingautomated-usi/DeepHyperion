
from multiprocessing import synchronize
from random import randint, uniform
import random
import logging as log
from features import neg_words, pos_words
from utils import find_adjs, listToString, get_synonym, untokenize
from properties import INPUT_MAXLEN


# replace a word with its synonym
def apply_mutoperator1(text):    

    # print("*************** replace a word with its synonym")

    vector = text.split()
    syn = None

    count = 0
    while syn == None or syn == []:
        rand_index = random.randint(0, len(vector)-1)  
        syn = get_synonym(vector[rand_index])
        count += 1
        if count > 2000:
            log.info("no synonym found!")
            break

    if syn != None and syn != []:
        vector[rand_index] = syn 
        
        text = untokenize(vector)


    return text

# remove a sentence
def apply_mutoperator2(text):

    # print("*************** remove a sentence")

    vector = text.split('.')

    if len(vector) > 1:
        rand_index = random.randint(0, len(vector)-1)  

        vector[rand_index] = ''
        text = listToString(vector)
    else:
        log.info("No sentence to remove!")

    return text

# reorder the sentences
def apply_mutoperator3(text):

    # print("*************** reorder the sentences")

    vector = text.split('.')

    if len(vector) > 1:
        rand_index1 = random.randint(0, len(vector)-1)  
        rand_index2 = random.randint(0, len(vector)-1)  
        while rand_index1 == rand_index2:
            rand_index2 = random.randint(0, len(vector)-1)  
        temp = vector[rand_index1]
        vector[rand_index1] = vector[rand_index2]
        vector[rand_index2] = temp
        text = listToString(vector)
    else:
        log.info("Not enough sentence to reorder!")

    return text

# duplicate a sentence
def apply_mutoperator4(text):

    # print("*************** duplicate a sentence")

    vector = text.split('.')
    rand_index = random.randint(0, len(vector)-1)   

    text = text + " " + vector[rand_index] + "."
    return text


# replace a word with its synonym
def apply_mutoperator5(text):    

    # print("*************** add a synonym")

    vector, adjs_index = find_adjs(text)
    final_vector = []
    syn = None

    # check if we have any adj in sentence
    if len(adjs_index) > 0:
        count = 0
        while syn == None or syn == []:
            rand_index = random.choice(adjs_index)  
            syn = get_synonym(vector[rand_index][0])
            count += 1
            if count > 2000:
                log.info("no synonym found!")
                break

        if syn != None and syn != []:
            vector[rand_index] = (vector[rand_index][0] + " and " + syn, vector[rand_index][1])

            # traverse in the string  
            for ele in vector: 
                final_vector.append(ele[0])

            text = untokenize(final_vector)
    else:
        log.info("no adjective found!")

    return text


def mutate(text, operator_name):
    mutant_vector = text    
    if operator_name == 1:
        mutant_vector = apply_mutoperator1(text)
#   elif operator_name == 2:        
#        mutant_vector = apply_mutoperator2(text)  
#    elif operator_name == 3:        
#        mutant_vector = apply_mutoperator3(text)
    elif operator_name == 4:        
        mutant_vector = apply_mutoperator4(text)
    elif operator_name == 5:
        mutant_vector = apply_mutoperator5(text)
    return mutant_vector
