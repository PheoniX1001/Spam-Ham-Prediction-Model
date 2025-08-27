#Importing Necessary Libraries

import pandas as pd
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Main cleaning process:
def clean(text):
    #removing punctuations
    punc_removed="".join([char for char in text if char not in string.punctuation])

    #tokenizing
    tknwords=nltk.word_tokenize(punc_removed)

    #removing stopwords (for eg 'is', 'the', 'a', 'an')
    stopwors= set(stopwords.words('english'))
    stop_removed=[word for word in tknwords if word.lower() not in stopwors]

    #stemming (reducing -ing, -ed to original for eg running to run)
    stemmer=PorterStemmer()
    stmd=[stemmer.stem(word) for word in stop_removed]
    final=" ".join(stmd)

    return final

ds=pd.read_csv('/home/sulav/Desktop/python/plt/spam.csv', encoding='latin1')
ds.rename(columns={'v1':'class','v2':'msg'},inplace=True)
ds.drop(columns={'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'},inplace=True)

ds_hams=ds[ds['class']=='ham']
ds_spams=ds[ds['class']=='spam']

#Balancing the dataset for model (no.of hams and spams are selected to be nearly equal)
tbr=ds_hams.sample(760,random_state=223)
df_balanced=pd.concat([ds_spams,tbr])
df_balanced['msg']=df_balanced['msg'].apply(clean)

y=df_balanced['class']
X=df_balanced['msg']

#Splitting dataset into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=648)
vct=TfidfVectorizer()
X_train_vct=vct.fit_transform(X_train)
X_test_vct=vct.transform(X_test)

#Creating prediction model
model=LogisticRegression()
model.fit(X_train_vct,y_train)

#New message input to check if spam or ham
nu=input("Enter a message: ")
clean_nu=clean(nu)
vect_nu=vct.transform([clean_nu])
prediction_Nu=model.predict(vect_nu)

if prediction_Nu[0]=='spam':
	print('spam')
else:
	print('ham')

y_prd=model.predict(X_test_vct)
acc=accuracy_score(y_test,y_prd)
print(f"Accuracy Score:{acc*100:.3f}%")



