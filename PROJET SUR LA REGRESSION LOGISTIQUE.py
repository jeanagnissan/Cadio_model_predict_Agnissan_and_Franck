#!/usr/bin/env python
# coding: utf-8

# # Projet IA et ML IGS
# 
# Importation des Bibliothèques utilisées

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,precision_score


# # Importation de nos Données

# In[41]:


data =pd.read_excel("C:\\Users\\DOROTHE\\Documents\\Ababa\\Guy\\Coeur1.xlsx")
df = data.copy()


# In[42]:


df=data.copy()


# In[43]:


df.head()


# Visualisation des 5 prémières lignes

# In[44]:


df.tail()


# Visualisation des 5 dernieres lignes

# In[45]:


print('Shape of data',df.shape)


# Dimension des lignes et colonnes

# In[46]:


df.info()


# Information générale sur les données

#  # Vérification des Doublons, Données Manquantes et constantes

# In[6]:


df.duplicated().sum()


# Aucun doublons

# In[31]:


df.isna().sum()/len(df)*100


# Et aucune données manquantes

# In[39]:


print(df.nunique())


# # Recodages des variables qualitatives et Normalisation de nos données sauf celui du Coeur

# In[9]:


var_numer=df._get_numeric_data().columns
var_qual=list(set(df.columns) - set(var_numer))


# In[11]:


def recoder(serie):
    return serie.astype('category').cat.codes
def codage(df):
    for  i in df.select_dtypes("object").columns:
        df[i]=recoder(df[i])
    return df
codage(df)


# In[12]:


def lisation(df):
    for col in var_numer:
        if col=='CŒUR':
            pass
        else:
             
            df[col]=df[col]/df[col].max()
    return df
lisation(df)


# # Décompositon en deux variables X et Y pour notre modèle de regression Logistique

# In[13]:


X=df.drop("CŒUR",axis=1)
Y=df["CŒUR"]


# # Découpage de nos données en test et entrainement

# In[14]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=5)


# In[48]:


print('Shape of X_train',X_train.shape)
print('Shape of X_test',X_test.shape)
print('Shape of Y_train',Y_train.shape)
print('Shape of Y_test',X_test.shape)


# # Création de notre modèle

# In[19]:


logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, Y_train)


# # Prédiction et la probabilité de notre prédiction(Score)

# In[21]:


prediction=logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, Y_test)
print(score)


# Notre modèle de régression nous donne une bonne prédiction, estimé à environ 0.87

#  # Précision et Sensiblité

# In[23]:


precision = precision_score(Y_test, prediction)
recall = recall_score(Y_test, prediction)
 
print('Precision: ',precision)
print('Recall: ',recall)


# In[24]:


predict_value=pd.Series(logisticRegr.predict(X_test),name="prediction")
df_confusion = pd.crosstab(Y_test, predict_value)
df_confusion


# # Matrice de confusion

# In[25]:


plt.figure(figsize=(23, 6))
sns.heatmap(X.corr(),
            square=True, linewidths=.4,annot=True)


# In[ ]:




