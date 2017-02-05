
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# ## Bibliotecas utilizadas

# In[1]:

from collections import namedtuple
from numbers import Number

import pandas as pd


# ## Lendo o arquivo e exibindo suas extremidades

# In[2]:

csv_train = pd.read_csv('datasets/train.csv')
csv_train.head()


# In[3]:

csv_train.tail()


# ## Criando um novo dataframe
# Esse novo dataframe terá apenas as colunas úteis para modelagem. Serão removidos o nome e o número do ticket, PassengerId será colocado como índice e as variáveis Sex e Embarked serão transformadas em dummy

# In[19]:

train = csv_train.copy()  # Cria uma cópia do dataframe
train.set_index('PassengerId', inplace=True)  # Define PassengerId como novo índice das linhas
dummies = pd.get_dummies(train.loc[:, ['Sex', 'Embarked']])  # transforma Sex e Embarked em variáveis dummy
train = pd.concat([train, dummies], axis=1)  # Concatena train e dummies
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)  # Remove colunas inúteis


# # Criando a variável CabinType

# In[20]:

def gera_CabinType(cabin):
    if isinstance(cabin, Number):
        return 'X'
    elif ' ' in cabin:  # Mais de uma cabine
        cabins = cabin.split()  # Separa todas as cabines em uma lista
        cabins = [c[0] for c in cabins]  # Pega o primeiro caractere de cada cabine
        return ''.join(set(cabins))  # Remove duplicatas e junta todos os CabinTypes
    else:
        return cabin[0]

CabinTypes = train['Cabin'].apply(gera_CabinType)
CabinTypes.name = 'CabinType'
train = pd.concat([train, CabinTypes], axis=1)
train.head()


# In[21]:

print(train.groupby('CabinType').apply(lambda x: x.Fare.min()))
print(train.groupby('CabinType').apply(lambda x: x.Fare.mean()))
print(train.groupby('CabinType').apply(lambda x: x.Fare.max()))


# ## Analisando valores missing em Cabin
# Comparando o preço e a classe dos passageiros sem número de cabine

# In[7]:

PreçosCabines = namedtuple('PreçosCabines', 'min méd max')
com_cabine = train.loc[train.Cabin.notnull(), :]
sem_cabine = train.loc[train.Cabin.isnull(), :]
preços_sem_cabine = PreçosCabines(sem_cabine.Fare.min(),
                                  sem_cabine.Fare.mean(),
                                  sem_cabine.Fare.max())
preços_com_cabine = PreçosCabines(com_cabine.Fare.min(),
                                  com_cabine.Fare.mean(),
                                  com_cabine.Fare.max())
print(f'Preços dos tickets dos passageiros sem cabine: {preços_sem_cabine}')
print(f'Preços dos tickets dos passageiros com cabine: {preços_com_cabine}')

train.loc[train.Fare > 512, :]


# ## Próximo passo: prever o tipo de cabine
