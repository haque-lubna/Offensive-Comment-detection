from django.shortcuts import render
from django.http import HttpResponse
import json


import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def check_profanity(sentence):
    df = pd.read_csv('./bangla_comment.csv')
    # df = pd.read_csv('./all_reduce.csv')
    df_names = df
    df_names.label.replace({'positive':1,'offensive':0},inplace = True)
    df_names.label.replace({'poitive':1,'offfensive':0},inplace = True)
    df_names.label.replace({'positivr':1,'offensive ':0},inplace = True)
    df_names.label.replace({'positive ':1,'offfensive':0},inplace = True)
    df_names.label.replace({'positive. ':1,'offensive':0},inplace = True)
    df_names.label.replace({'positive.':1,'offfensive':0},inplace = True)
    df_names.label.unique()
    Xfeatures = df_names['comment']
    cv = CountVectorizer()
    X = cv.fit_transform(Xfeatures)
    cv.get_feature_names()
    X
    y = df_names.label
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test, y_test)


    # new dataset
    # df = pd.read_csv('./bangla_comment.csv')
    # # df = pd.read_csv('./all_reduce.csv')
    # df_names = df
    # df_names.label.replace({'positive':1,'offensive':0},inplace = True)
    # df_names.label.replace({'poitive':1,'offfensive':0},inplace = True)
    # df_names.label.replace({'positivr':1,'offensive ':0},inplace = True)
    # df_names.label.replace({'positive ':1,'offfensive':0},inplace = True)
    # df_names.label.replace({'positive. ':1,'offensive.':0},inplace = True)
    # df_names.label.replace({'positive.':1,'offfensive':0},inplace = True)
    # df_names.label.replace({'p':1,'offfensive':0},inplace = True)
    # #english
    # df_names.label.replace({'positive ':1,'offensive ':0},inplace = True)
    # df_names.label.replace({'positivo':1,'offensive':0},inplace = True)
    # df_names[df_names.label.notnull()]
    # df_names.label.unique()
    # df_names.shape[0]
    # # # df_names.isnull().sum()
    # df_names = df_names[df_names.comment.notnull()]
    # df_names.shape[0]
    # df_names.head(5)
    # df_names.shape[0]
    # df_names.label.unique()
    
    # #Step 2 : Feature Extraction. Also known as vectorizing data. Basically digitalizing it in meaningful ways.
    # from sklearn import model_selection, preprocessing
    # from sklearn.model_selection import train_test_split
    # from sklearn.feature_extraction.text import CountVectorizer
    # #everything needs to be digitalized. Labels and Texts.
    # #labels digitalized here in Y axis
    # encoder = preprocessing.LabelEncoder()
    # y = encoder.fit_transform(df['label'].values.astype('U'))


    # #Posts digitalized here in x-axis
    # cv = CountVectorizer()
    # X_data_full_tfidf = cv.fit_transform(df['comment'].values.astype('U'))
    # # tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
    # # tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2))
    # # X_data_full_tfidf = tfidf.fit_transform(df['post'])


    # X_train, X_test, y_train, y_test = train_test_split(X_data_full_tfidf, y, test_size=0.4 , random_state = 52)
    # # # #Step 3: train classifier

    # # #with a naive bayes
    # # from sklearn.naive_bayes import MultinomialNB
    # # from sklearn.svm import SVC


    # # clf = MultinomialNB().fit(X_train, y_train)
    # # clf.score(X_test, y_test)
    # # print("naive bayes accuracy is")

    # # # #step 4: evaluate classifier

    # # print(clf.score(X_test, y_test))

    # # # #with a support vector machine
    # from sklearn.svm import SVC
    # clf = SVC(kernel='linear')
    # clf.fit(X_train, y_train)
    # clf.score(X_test, y_test)
    
    if len(sys.argv)>1:
        sample_name = [sentence]
    else:
        sample_name = ["আমার"]
    vect = cv.transform(sample_name).toarray()
    return "{}".format(clf.predict(vect))

def index(request):
    body_unicode = request.body.decode('utf-8')
    body_data = json.loads(body_unicode)
    data = body_data
    sentence = data['sentence']
    ret = check_profanity(sentence)
    return HttpResponse(ret)