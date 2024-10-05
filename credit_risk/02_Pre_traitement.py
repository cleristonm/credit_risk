#!/usr/bin/env python
# coding: utf-8

# In[291]:


import pandas as pd

df = pd.read_csv('../data/raw/credit_risk_dataset.csv')


# # Outliers - Suppression des valeurs aberrantes

# In[293]:


# 1. Moyenne
# 2. Mode
# 3. Supprimer

mode_remplacement = 1


# In[294]:


import numpy as np
import pandas as pd

def valeurs_aberrantes(col):
    print(f'\nChamp: {col}')
    # Supprimer les valeurs nulles
    colonne_sans_nuls = df[col].dropna()
    
    # Convertir en numérique
    colonne_sans_nuls = pd.to_numeric(colonne_sans_nuls)
    mean = colonne_sans_nuls.mean()
    
    # Calculer les quartiles
    q1 = colonne_sans_nuls.quantile(0.25)
    q3 = colonne_sans_nuls.quantile(0.75)
    
    # Calculer l'intervalle interquartile
    iqr = q3 - q1
    
    # Définir les limites pour les valeurs aberrantes
    limite_inférieure = q1 - 1.5 * iqr
    limite_supérieure = q3 + 1.5 * iqr
    
    # Lister les valeurs aberrantes
    valeur_aberrantes = df[col][(df[col].notna()) & 
                                           ((pd.to_numeric(df[col]) < limite_inférieure) | 
                                            (pd.to_numeric(df[col]) > limite_supérieure))]
    
    print(f'Valeur moyenne: {mean}')
    print(f'Min valeur aberrante: {min(valeur_aberrantes)}')
    print(f'Max valeur aberrante: {max(valeur_aberrantes)}')
    


# In[295]:


valeurs_aberrantes('person_age')
valeurs_aberrantes('person_emp_length')
valeurs_aberrantes('cb_person_cred_hist_length')


# Nous pouvons constater que les champs person_age et person_emp_length présentent des valeurs aberrantes. 
# Définissons ces valeurs comme nulles

# In[297]:


df.loc[df['person_age'] >= 80, 'person_age'] = None
df.loc[df['person_emp_length'] >= 60, 'person_emp_length'] = None


# In[298]:


def traitement_na(champ, df):
    if (mode_remplacement == 1):
        df[champ] = df[champ].fillna(df[champ].mean())
    elif (mode_remplacement == 2):
        df[champ] = df[champ].fillna(df[champ].mode()[0])
    elif (mode_remplacement == 3):
        num_lignes_avant = len(df)
        df = df[df[champ].notna()]
        print(f'Champ: {champ} - Lignes supprimées {num_lignes_avant - len(df)}')
    return df

df = traitement_na('person_age', df)
df = traitement_na('person_emp_length', df)
df = traitement_na('loan_int_rate', df)


# # Enlever des doublons
# 
# Les doublons peuvent créer un biais dans les résultats de l'analyse 

# In[300]:


df = df.drop_duplicates(keep='first')


# # Encodage

# ## One Hot Encoder - Loan_intent

# In[303]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

encoded_loan_intent = encoder.fit_transform(df[['loan_intent']]).toarray()

encoded_df = pd.DataFrame(encoded_loan_intent, columns=encoder.get_feature_names_out(['loan_intent']), 
                          index=df.index)

df = pd.concat([df, encoded_df], axis=1)


# ## One Hot Encoder - person_home_ownership

# In[305]:


encoded_person_home_ownership = encoder.fit_transform(df[['person_home_ownership']]).toarray()

encoded_df = pd.DataFrame(encoded_person_home_ownership, columns=encoder.get_feature_names_out(['person_home_ownership']), 
                          index=df.index)

df = pd.concat([df, encoded_df], axis=1)


# ## OrdinalEncoder - loan_grade_encoded

# In[307]:


from sklearn.preprocessing import OrdinalEncoder

ordre = np.sort(df['loan_grade'].unique())

# Création de l'Ordinal Encoder
oe = OrdinalEncoder(categories=[ordre])

# Transformation de la colonne
df['loan_grade_encoded'] = oe.fit_transform(df[['loan_grade']])


# In[308]:


from sklearn.preprocessing import LabelEncoder


# Création d'un Label Encoder
le = LabelEncoder()

# Fit et transform
df['cb_person_default_on_file_encoded'] = le.fit_transform(df['cb_person_default_on_file'])


# ## Scaler 

# In[310]:


numeric_features = ["person_age","person_income","person_emp_length","loan_int_rate", "loan_amnt", "loan_percent_income", "cb_person_cred_hist_length"]
categorical_features = ["person_home_ownership","loan_grade", "loan_intent","cb_person_default_on_file", "cb_person_default_on_file"]


# In[311]:


###TODO VERIFIER RobustScaler POUR LES OUTLIERS 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])


# In[312]:


df.to_csv('../data/processed/credit_risk_dataset_processed.csv', index=False)

