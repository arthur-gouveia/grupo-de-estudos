
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# ## Bibliotecas utilizadas

# In[1]:

import pandas as pd


# ## Lendo o arquivo e exibindo suas extremidades

# In[2]:

csv_train = pd.read_csv('datasets/train.csv')
csv_train.head()


# In[3]:

csv_train.tail()


# ## Criando um novo dataframe
# Esse novo dataframe terá apenas as colunas úteis para modelagem. Serão removidos o nome e o número do ticket, PassengerId será colocado como índice e as variáveis Sex e Embarked serão transformadas em dummy

# In[12]:

train = csv_train.copy()  # Cria uma cópia do dataframe
train.set_index('PassengerId', inplace=True)  # Define PassengerId como novo índice das linhas
dummies = pd.get_dummies(train.loc[:, ['Sex', 'Embarked']])  # transforma Sex e Embarked em variáveis dummy
train = pd.concat([train, dummies], axis=1)  # Concatena train e dummies
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)  # Remove colunas inúteis
train.head()


# In[9]:

dummies = pd.get_dummies(train.loc[:, ['Sex', 'Embarked']])


# In[11]:

pd.concat([train, dummies], axis=1)


# In[8]:

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1)

