
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# ## Bibliotecas utilizadas

# In[1]:

from numbers import Number

import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LinearRegression

get_ipython().magic('matplotlib inline')


# ## Lendo o arquivo e exibindo suas extremidades

# In[2]:

csv_train = pd.read_csv('datasets/train.csv')
csv_train.head()


# In[3]:

csv_train.tail()


# In[4]:

csv_train.describe()


# ## Criando um novo dataframe
# Esse novo dataframe terá apenas as colunas úteis para modelagem. Serão removidos o nome e o número do ticket, PassengerId será colocado como índice e as variáveis Sex e Embarked serão transformadas em dummy

# In[5]:

train = csv_train.copy()  # Cria uma cópia do dataframe
train.set_index('PassengerId', inplace=True)  # Define PassengerId como novo índice das linhas
dummies = pd.get_dummies(train.loc[:, ['Sex', 'Embarked']])  # transforma Sex e Embarked em variáveis dummy
train = pd.concat([train, dummies], axis=1)  # Concatena train e dummies
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)  # Remove colunas inúteis


# ## Preenchendo os valores missing de idade

# In[6]:

X = train.loc[train.Age.notnull(), :].copy()
Y = X.Age
X.drop(['Cabin', 'Age'], axis=1, inplace=True)

linreg = LinearRegression()
linreg.fit(X, Y)


X2 = train.loc[train.Age.isnull(),:].copy()
X2.drop(['Cabin', 'Age'], axis=1, inplace=True)
s = pd.Series(linreg.predict(X2))
s.index = train.Age[train.Age.isnull()].index
train.Age.fillna(value=s, inplace=True)
train.describe()


# # Criando a variável cabin_type
# O número da cabine, por si só, não é uma informação muito útil. Porém a letra inicial da cabine pode ser um indicativo da localização da mesma e isso pode indicar o 'status' do passageiro que ocupa aquela cabine. Dessa forma vamos isolar a primeira letra da cabine e chamá-la de CabinType

# In[7]:

def gera_CabinType(cabin):
    if isinstance(cabin, Number):
        return 'X'
    elif ' ' in cabin:  # Mais de uma cabine
        cabins = cabin.split()  # Separa todas as cabines em uma lista
        cabins = [c[0] for c in cabins]  # Pega o primeiro caractere de cada cabine
        return ''.join(set(cabins))  # Remove duplicatas e junta todos os CabinTypes
    else:
        return cabin[0]

cabin_types = train['Cabin'].apply(gera_CabinType)
cabin_types.name = 'Cabin_Type'
train = pd.concat([train, cabin_types], axis=1)
train.head()


# ## Analisando valores missing em Cabin
# Comparando o preço e a classe dos passageiros sem número de cabine (ou com CabinType 'X')

# In[8]:

def gera_preços(df):
    return pd.DataFrame({'Valor mínimo': df.Fare.min(),
            'Valor médio': df.Fare.mean(),
            'Valor máximo': df.Fare.max(),
            }, index=[len(df.Fare)])

train.groupby('Cabin_Type').apply(func=gera_preços)


# In[9]:

com_cabine = train.loc[train.Cabin.notnull(), :]
com_cabine.boxplot('Fare', by='Cabin_Type', figsize=(12, 8))


# **É possível ver que há evidências de uma relação entre CabinType e Fare. Vou testar isso com uma ANOVA**

# In[10]:

mod = ols('Fare ~ Cabin_Type', data=com_cabine).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# Há uma diferença significativa das médias de ```Fare``` para os diversos grupos de ```CabinType```. Dessa forma é possível gerar um modelo que encontre o ```CabinType``` em função do ```Fare```

# ## Prevendo o tipo de cabine

# In[37]:

X = com_cabine.drop(['Cabin_Type', 'Cabin'], axis=1)
y = com_cabine.Cabin_Type.copy()

dtc = tree.DecisionTreeClassifier()
dtc.fit(X, y)

train.Cabin_Type = dtc.predict(train.drop(['Cabin_Type', 'Cabin'], axis=1))
train

