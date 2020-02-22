# -*- coding: utf-8 -*-

import numpy as np
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
np.random.seed(1337)
import json, re, nltk, string
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, Concatenate, concatenate
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity 
from keras.utils import to_categorical

# Web crawler 
from bs4 import BeautifulSoup
from lxml import html
import xml
import requests

# ========================================================================================
# Dataset Absolute Path
# ========================================================================================

open_bugs_json = 'C:\dataset\TestData.json'
# open_bugs_json = 'gdrive/My Drive/dataset/TestData.json'
closed_bugs_json = 'C:\dataset\TrainData.json'
# closed_bugs_json = 'gdrive/My Drive/dataset/TrainData.json'
web_data_address = 'gdrive/My Drive/test/webData.json'


# ========================================================================================
# Initializing Hyper parameter
# ========================================================================================
# 1. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5

# 2. Classifier hyperparameters

numCV = 1
max_sentence_length = 50
min_sentence_length = 15
rankK = 10
batch_size = 128
myEpochs = 1

# ========================================================================================
# Preprocess the open bugs, extract the vocabulary and learn the word2vec representation
# ========================================================================================
with open(open_bugs_json) as data_file:
    data = json.load(data_file, strict=False)

all_data = []
for item in data:
    # 1. Remove \r
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')
    # 2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                          current_desc)
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title = re.sub(r'(\w+)0x\w+', '', current_title)
    # 5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    # 6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    # 7. Strip trailing punctuation marks
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]
    # 8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = list(filter(None, current_data))
    all_data.append(current_data)

# print (all_data[:5])
# print (len(all_data))

# Learn the word2vec model and extract vocabulary
wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec,
                         window=context_window_word2vec)
vocabulary = wordvec_model.wv.vocab
vocab_size = len(vocabulary)

# ========================================================================================
# Preprocess the closed bugs, using the extracted the vocabulary
# ========================================================================================
with open(closed_bugs_json) as data_file:
    data = json.load(data_file, strict=False)

all_data = []
all_owner = []
for item in data:
    # 1. Remove \r
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')
    # 2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                          current_desc)
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title = re.sub(r'(\w+)0x\w+', '', current_title)
    # 5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    # 6. Tokenize
    # A sentence or data can be split into words using the method word_tokenize()
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    # 7. Strip punctuation marks
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]
    # 8. Join the lists
    current_data = current_title_filter + current_desc_filter
    ###########################
    # current_data = filter(None, current_data)
    current_data = list(filter(None, current_data))
    ###########################
    all_data.append(current_data)
    all_owner.append(item['owner'])

# Test
# print('Preprocess the closed bugs:')
# print(all_data[:5])
# print(all_owner[:200])
# all_data[:5]

# ========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
# ========================================================================================
with open(web_data_address) as f:
    web_data = json.load(f)

totalLength = len(all_data)
#print(totalLength)
splitLength = totalLength // numCV
#print(splitLength)

for i in range(1, numCV + 1):
    # Split cross validation set
    print (i)
    train_data = all_data[:i * splitLength - 1]
    test_data = all_data[(i-1) * splitLength:i * splitLength - 1]
    train_owner = all_owner[:i * splitLength - 1]
    test_owner = all_owner[(i-1) * splitLength:i * splitLength - 1]

    # Remove words outside the vocabulary
    updated_train_data = []
    updated_train_data_length = []
    updated_train_owner = []
    final_test_data = []
    final_test_owner = []
    for j, item in enumerate(train_data):
        current_train_filter = [word for word in item if word in vocabulary]
        if len(current_train_filter) >= min_sentence_length:
            updated_train_data.append(current_train_filter)
            updated_train_owner.append(train_owner[j])

    for j, item in enumerate(test_data):
        current_test_filter = [word for word in item if word in vocabulary]
        if len(current_test_filter) >= min_sentence_length:
            final_test_data.append(current_test_filter)
            final_test_owner.append(test_owner[j])

            # Remove data from test set that is not there in train set
    train_owner_unique = set(updated_train_owner)
    test_owner_unique = set(final_test_owner)
    unwanted_owner = list(test_owner_unique - train_owner_unique)
    updated_test_data = []
    updated_test_owner = []
    updated_test_data_length = []
    for j in range(len(final_test_owner)):
        if final_test_owner[j] not in unwanted_owner:
            updated_test_data.append(final_test_data[j])
            updated_test_owner.append(final_test_owner[j])

    unique_train_label = list(set(updated_train_owner))
    classes = np.array(unique_train_label)

    # Create train and test data for deep learning + softmax
    X_train = np.empty(shape=[len(updated_train_data), max_sentence_length, embed_size_word2vec], dtype='float32')
    Y_train = np.empty(shape=[len(updated_train_owner), 1], dtype='int32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
    for j, curr_row in enumerate(updated_train_data):
        sequence_cnt = 0
        for item in curr_row:
            if item in vocabulary:
                X_train[j, sequence_cnt, :] = wordvec_model.wv[item]
                sequence_cnt = sequence_cnt + 1
                if sequence_cnt == max_sentence_length - 1:
                    break
        for k in range(sequence_cnt, max_sentence_length):
            X_train[j, k, :] = np.zeros((1, embed_size_word2vec))
        Y_train[j, 0] = unique_train_label.index(updated_train_owner[j])

    X_test = np.empty(shape=[len(updated_test_data), max_sentence_length, embed_size_word2vec], dtype='float32')
    Y_test = np.empty(shape=[len(updated_test_owner), 1], dtype='int32')

    # Store the tested sentence or content of testing data set
    testedContent = np.empty(shape=[len(updated_test_data), max_sentence_length], dtype='U32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
    for j, curr_row in enumerate(updated_test_data):
        sequence_cnt = 0
        for item in curr_row:
            if item in vocabulary:
                X_test[j, sequence_cnt, :] = wordvec_model.wv[item]
                # Assign the value of item to tested content
                testedContent[j][sequence_cnt] = item
                sequence_cnt = sequence_cnt + 1
                if sequence_cnt == max_sentence_length - 1:
                    break
        for k in range(sequence_cnt, max_sentence_length):
            X_test[j, k, :] = np.zeros((1, embed_size_word2vec))
        Y_test[j, 0] = unique_train_label.index(updated_test_owner[j])

    y_train = to_categorical(Y_train, len(unique_train_label))
    y_test = to_categorical(Y_test, len(unique_train_label))

    # Construct the deep learning model
    sequence = Input(shape=(max_sentence_length, embed_size_word2vec), dtype='float32')
    forwards_1 = LSTM(1024)(sequence)
    after_dp_forward_4 = Dropout(0.20)(forwards_1)
    backwards_1 = LSTM(1024, go_backwards=True)(sequence)
    after_dp_backward_4 = Dropout(0.20)(backwards_1)
    merged = concatenate([after_dp_forward_4, after_dp_backward_4], axis=-1)
    after_dp = Dropout(0.5)(merged)
    output = Dense(len(unique_train_label), activation='softmax')(after_dp)
    model = Model(inputs=sequence, outputs=output)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

    # fit the model
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=myEpochs)

    predict = model.predict(X_test)
    accuracy = []
    sortedIndices = []
    pred_classes = []
    for ll in predict:
        sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))

    for myMethods in range(1, 3):
        # Deep learning approach
        if myMethods == 1:
            print('=====Deep learning method=====')
            for k in range(1, rankK + 1):
                id = 0
                trueNum = 0
                lableOwner = []
                lableOwnerId = []
                for sortedInd in sortedIndices:
                    pred_classes.append(classes[sortedInd[:k]])
                    if Y_test[id] in sortedInd[:k]:
                        trueNum += 1

                    # store correct owner
                    else:
                        lableOwner.append(classes[Y_test[id]])
                        lableOwnerId.append(id)

                    id += 1
                accuracy.append((float(trueNum) / len(predict)) * 100)
            print('Test accuracy:', accuracy[-1])

        #     # print error example
        #     for i in range (0,20):
        #         print(i)
        #         print('Testing sentence:   ', testedContent[lableOwnerId[i]])
        #         print('Correct sentence:   ', lableOwner[i])
        #         print('Predict sentence:   ', pred_classes[lableOwnerId[i]])
        #         print()

        #     train_result = hist.history
        # print(train_result)

        # Deep learning + rule engine approach
        else:
            print('=====Deep learning + rule engine method=====')
            for k in range(1, rankK + 1):
                id = 0
                user_index = 0
                runs = 100
                trueNum = 0
                lableOwner = []
                lableOwnerId = []
                for sortedInd in sortedIndices:
                    pred_classes.append(classes[sortedInd[:k]])
                    if Y_test[id] in sortedInd[:k]:
                        trueNum += 1

                    # store error data
                    elif user_index < runs and k == rankK:
                        print("Wrong prediction, rule engine initiate")
                        print("Correct Owner:::", classes[Y_test[id]])
                        print("Tensorflow predicted owners from:::", classes[sortedInd[:k]])

                        lableOwner.append(classes[Y_test[id]])
                        lableOwnerId.append(id)
                        # get web_data
                        owner = ' '.join(map(str, lableOwner[user_index]))
                        unique_result = []
                        if owner in web_data:
                            unique_result = web_data[owner]
                        # Get test set
                        test = testedContent[lableOwnerId[user_index]]

                        # Get overlap percentage
                        overlap = set(unique_result) & set(list(filter(None, test)))
                        percentage_overlap = float(len(overlap) / len(test))

                        # The customize rule
                        bar_percentage = 0.3
                        rule_flag = True if percentage_overlap > bar_percentage else False
                        if rule_flag:
                            trueNum += 1
                        print("Test content::: ", test)
                        print("Web data:::: ", unique_result)
                        print("Overlap content:::", overlap)

                        # print('user_index:  ', user_index)
                        print('overlap percentage:::   ', percentage_overlap)
                        # print('rule result:::  ', rule_flag)
                        print('  ')
                        user_index += 1

                    else:
                        lableOwner.append(classes[Y_test[id]])
                        lableOwnerId.append(id)

                    id += 1
                accuracy.append((float(trueNum) / len(predict)) * 100)
            print('Test accuracy:', accuracy[-1])

            #     print error example
            #     for i in range (0,20):
            #         print(i)
            #         print('Testing sentence:   ', testedContent[lableOwnerId[i]])
            #         print('Correct sentence:   ', lableOwner[i])
            #         print('Predict sentence:   ', pred_classes[lableOwnerId[i]])
            #         print()

            train_result = hist.history
            # print(train_result)

    del model

# ========================================================================================
# Split cross validation sets and perform baseline classifiers
# ========================================================================================

totalLength = len(all_data)
#print(totalLength)
splitLength = totalLength // numCV
#print(splitLength)
#print (len(all_data))
#print (len(all_owner))
print('=====Naive Bayes method=====')

for i in range(1, numCV + 1):
    # Split cross validation set
    print (i)
    train_data = all_data[:i * splitLength - 1]
    test_data = all_data[(i-1) * splitLength:i * splitLength - 1]
    train_owner = all_owner[:i * splitLength - 1]
    test_owner = all_owner[(i-1) * splitLength:i * splitLength - 1]

    # Remove words outside the vocabulary
    updated_train_data = []
    updated_train_data_length = []
    updated_train_owner = []
    final_test_data = []
    final_test_owner = []


    for j, item in enumerate(train_data):
        current_train_filter = [word for word in item if word in vocabulary]
        if len(current_train_filter) >= min_sentence_length:
            updated_train_data.append(current_train_filter)
            updated_train_owner.append(train_owner[j])

    for j, item in enumerate(test_data):
        current_test_filter = [word for word in item if word in vocabulary]
        if len(current_test_filter) >= min_sentence_length:
            final_test_data.append(current_test_filter)
            final_test_owner.append(test_owner[j])

            # Remove data from test set that is not there in train set
    train_owner_unique = set(updated_train_owner)
    test_owner_unique = set(final_test_owner)
    unwanted_owner = list(test_owner_unique - train_owner_unique)
    updated_test_data = []
    updated_test_owner = []
    updated_test_data_length = []

    for j in range(len(final_test_owner)):
        if final_test_owner[j] not in unwanted_owner:
            updated_test_data.append(final_test_data[j])
            updated_test_owner.append(final_test_owner[j])

    train_data = []
    for item in updated_train_data:
        train_data.append(' '.join(item))

    test_data = []
    for item in updated_test_data:
        test_data.append(' '.join(item))

    vocab_data = []
    for item in vocabulary:
        vocab_data.append(item)

    # Extract tf based bag of words representation
    tfidf_transformer = TfidfTransformer(use_idf=False)
    count_vect = CountVectorizer(min_df=1, vocabulary=vocab_data, dtype=np.int32)

    train_counts = count_vect.fit_transform(train_data)
    train_feats = tfidf_transformer.fit_transform(train_counts)
    #print (train_feats.shape)

    test_counts = count_vect.transform(test_data)
    test_feats = tfidf_transformer.transform(test_counts)
    #print (test_feats.shape)
    #print ("=======================")

    # perform classification

    classifierModel = MultinomialNB(alpha=0.01)
    classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
    predict = classifierModel.predict_proba(test_feats)
    classes = classifierModel.classes_

    accuracy = []
    sortedIndices = []
    pred_classes = []


    for ll in predict:
        sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))

    for k in range(1, rankK + 1):
        id = 0
        trueNum = 0
        for sortedInd in sortedIndices:
            pred_classes.append(classes[sortedInd[:k]])
            if(id < len(Y_test)):
                if Y_test[id] in sortedInd[:k]:
                    trueNum += 1
                    #pred_classes.append(classes[sortedInd[:k]])
                id += 1
        accuracy.append((float(trueNum) / len(predict)) * 100)
    print ('Naive Bayes accuracy:', accuracy[2])
