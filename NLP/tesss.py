# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
from pprint import pprint
import spacy
from spacy import displacy
from nltk.corpus import stopwords
from collections import Counter
import en_core_web_sm

from sklearn.model_selection import train_test_split

#
from keras import optimizers
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
import pandas as pd
import numpy as np
import re
from keras.utils import plot_model
from numpy import array
from numpy import asarray
from numpy import zeros
import datetime

#

    #sudah bisa, linkny ini ya, https://stackoverflow.com/questions/17280534/prettyprint-to-a-file/17280610

    #NEXT## input ke dataframe  data[["date_val","time_val","people_val","org_val"]]
data = pd.read_csv("/Users/glendr/Documents/Temp Internal/CodeNLP/datasetTrain/data3.csv", delimiter=',', encoding="ISO-8859-1")
# data= pd.read_csv("/Users/glendr/Documents/Temp Internal/CodeNLP/datasetTrain/")#dataset
# data.head()
# print(data.head())

datainput=data[["dateval","timeval","peopleval","orgval"]]
# datainput.head()
# print(datainput.head(41))
from keras.utils import to_categorical
dataresulted=data["predict"]
# print(dataresulted.head())

y= dataresulted.values
# y= to_categorical(datainput)
X = datainput.values #tipe num

#split dataset 
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.30,random_state=15)
#yang diambil Xtrain ma Ytrain buat diTest(diFit)



#https://keras.io/models/sequential/ buat doc yang dibawah 1.Model 2.Compile+optimizer 3.Train-test
#model
mymodel=Sequential()
#1 input ,1prediksi ,1output ---- layer
#isi perlayer harusnya 6-8-2
mymodel.add(Dense(20,input_dim=4))
# mymodel.add(Dense(20, activation='sigmoid'))
mymodel.add(Dense(1, activation='sigmoid')) #output 
#softmax
#sgd = optimizers.SGD(lr=0.1, clipvalue=0.5) adalah mlp 
##Before training a model, you need to configure the learning process, which is done via the compile method.
sgd = optimizers.SGD(lr=0.1, clipvalue=0.5)
mymodel.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
#binary_crossentropy,mean_squared_error,hinge,squared_hinge,categorical_hinge,categorical_crossentropy
#Mean squared error / kuadrat kesalahan atau rata-rata kuadrat dari estimator mengukur rata-rata kuadrat dari kesalahan-yaitu, perbedaan kuadrat rata-rata antara nilai estimasi dan nilai aktual. https://keras.io/losses/

##Train
# Train the model, iterating on the data in batches of 32 samples
# Train model dengan 6 epoch, Clearly, if you have a small number of epochs in your training, it would be poor and you would realize the effects of underfitting. On the other hand, if you train the network too much, it would 'memorize' the desired outputs for the training inputs (supposing a supervised learning)
mymodel.fit(xtrain,ytrain,epochs=5,batch_size=10,validation_split=0.30)



#Returns the loss value & metrics values for the model in test mode.
#verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
score=mymodel.evaluate(xtest,ytest,verbose=1)
# print("evaluate")

# print(mymodel.summary())

# import matplotlib.pyplot as plt

# pri=mymodel.fit(xtrain,ytrain,epochs=7,batch_size=10,validation_split=0.30)
# plt.plot(pri.history['acc'])
# plt.plot(pri.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train','test'], loc='upper left')
# plt.show()


#return numpy array!
# mymodel.predict(input,verbose=1)

def datamasuk(text):
    stop_words = set(stopwords.words('english'))
    nlp = en_core_web_sm.load()
    # q='Hello. This is Beauty Hair. How can I help you? Hi. Can I make an appointment with Lucy, please? Is it for a hair treatment? No, it’s for a haircut and coloring. Oh, okay. When would you like to come in? Is there anything available after four on Tuesday? Is 6 pm okay? Yeah, that’s fine.  All right. What’s your name, please? Meghan George. Thanks.'
    q=text
    q= re.sub('\n'," ",q)
    q= q.lower()
    q=[w for w in q.split() if not w in stop_words]
    q=' '.join(q)
    print("\n"+q)
    text=nlp(q)
    
    d1=0
    d1text=""
    d2=0
    d2text1=""
    d2text=""
    d3=0
    d3text=""
    d4=0
    d4text=""
    
    for ent in text.ents:
            #
        #if else if ada =1 else 0, ex: "date": 19/09/11 ,"dateval": 1 . "date": ,"dateval": 0
        print(ent.text, ent.label_)
        if(ent.label_=="PERSON"):
            d1=1
            d1text=ent.text
        if(ent.label_=="DATE"):
            d2text=ent.text
            d2=1
            try:
                d2text1=datetime.datetime.strptime(ent.text,"%m/%d/%y")
                d2=1
            except ValueError  as err:
                pass
            # d2text=x
            # d2text=ent.text
        if(ent.label_=="TIME"):
            d3=1
            d3text=ent.text
        if(ent.label_=="ORG"):
            d4text=ent.text
            d4=1
        if(ent.label_=="GPE"):
            d4=1
            d4text=ent.text
        if(ent.label_=="FAC"):
            d4=1
            d4text=ent.text
    if(d2text1!=""):
        d2text=d2text1
    print("\n\nPerson: "+d1text+"\nDate: "+d2text+"\nTime: "+d3text+"\nother: "+d4text)
    print("\n\n")
    # print("\n\nPerson: "+d1+"\nDate: "+d2+"\nTime: "+d3+"\nother: "+d4)
    print("Next Section")

    # pprint([(X.text, X.label_) for X in text.ents])
    try:
        print("\n\n")
        dataf=pd.DataFrame({'dateval': [int(d2)],"timeval": [int(d3)],"peopleval": [int(d1)],"orgval":[ int(d4)]})
        print(dataf.head())

        try:
            f=open("result.txt","w+")
            if(d1text==""):
                d1text="No_Person"
            if(d2text==""):
                d2text="Tomorrow_nd"
            if(d3text==""):
                d3text="7:01"
            if(d4text==""):
                d4text="No_other_info"
            f.write(dat+";"+d2text+";"+d3text+";"+d1text+";"+d4text+";")
            f.close()
        except:
            print("error code: 3")
        return dataf   
    except:
        print("\nerror code: 1\n")

pre=""
try:
    f=open("chatdiscord.txt","r")
    dat=f.readline()
    text=dict()
    text=datamasuk(dat)
    a=mymodel.predict_on_batch(text)
    
    print("\n\n")
    print(a)
    print("\n\n")

    b=float(a)
    if(b>=0.55):
        pre="True"
        print("\tTrue")
    else:
        pre="False"
        print("\tFalse")
    f.close()
except:
    print("error code: 2")


try:
    f=open("result.txt","a")
    f.write(pre+";")
    f.close()
except:
    print("error code: 3.2")

# a=mymodel.predict(text,verbose=1)
#return Numpy array(s) of predictions.

#https://www.geeksforgeeks.org/numpy-array_str-in-python/
# b=np.array_str(a)
# print(b)










