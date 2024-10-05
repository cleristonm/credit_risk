#!/usr/bin/env python
# coding: utf-8

# In[205]:


#!pip install xgboost
import pandas as pd

df = pd.read_csv('../data/processed/credit_risk_dataset_processed.csv')


# In[206]:


colonnes_numeriques = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


# In[207]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import time
from tabulate import tabulate

# Fonction pour entraîner et évaluer le modèle
def entrainer_evaluer_modele(nom, modele, colonnes_a_enlever, description):
    colonnes_a_entrainer = list(set(colonnes_numeriques) - set(colonnes_a_enlever) - set(['loan_status']))
   
    X = df[colonnes_a_entrainer]
    
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    debut = time.time()
    modele.fit(X_train, y_train)
    previsions = modele.predict(X_test)
    fin = time.time()
    matrice_confusion = confusion_matrix(y_test, previsions)
        
    return {
        'Description': description,
        'Algorithme': nom,
        'Précision': precision_score(y_test, previsions),
        'Rappel': recall_score(y_test, previsions),
        'F1-score': f1_score(y_test, previsions),
        'Accuracy': accuracy_score(y_test, previsions),
        'Temps dexécution': fin - debut,
        'TP': matrice_confusion[1, 1],
        'FN': matrice_confusion[1, 0],
        'FP': matrice_confusion[0, 1],
        'TN': matrice_confusion[0, 0],
        'Modele': modele
    }


def executer_modeles(colonnes_a_enlever, description):
    # Dictionnaire des modèles
    modeles = {
        'Régression logistique': LogisticRegression(max_iter=1000),
        'Arbre de décision': DecisionTreeClassifier(max_depth=10),
        'Forêt aléatoire': RandomForestClassifier(max_depth=20),
        'KNN': KNeighborsClassifier(n_neighbors=10),
        'Naive Bayes': GaussianNB(),
        'XGBoost': xgb.XGBClassifier(
            colsample_bytree=0.8, gamma=0.1, learning_rate=0.1,
            max_depth=6, n_estimators=200, subsample=1, random_state=42)
    }
    
    # Entraîner et collecter les résultats
    resultats = [entrainer_evaluer_modele(nom, modele, colonnes_a_enlever, description) for nom, modele in modeles.items()]
    
    # Afficher les résultats
    tableau = [[modele['Algorithme'], 
                f"{modele['Précision']:.3f}", 
                f"{modele['Rappel']:.3f}", 
                f"{modele['F1-score']:.3f}", 
                f"{modele['Accuracy']:.3f}",
                f"{modele['Temps dexécution']:.2f}",
                f"{modele['TP']}", 
                f"{modele['FN']}", 
                f"{modele['FP']}", 
                f"{modele['TN']}"] 
               for modele in resultats]
    print(description)
    
    print(tabulate(tableau, 
                   headers=['Algorithme', 'Précision', 'Rappel', 'F1-score', 'Accuracy', 'Temps d\'exécution', 'TP', 'FN', 'FP', 'TN'], 
                   tablefmt='orgtbl'))
    print()
    return resultats


# In[208]:


model_1 = executer_modeles([], 'Toutes les colonnes numériques')
model_2 = executer_modeles(['cb_person_cred_hist_length'], 'Colonnes supprimées: cb_person_cred_hist_length')

#
model_3 = executer_modeles(['cb_person_cred_hist_length', 'loan_percent_income'], 'Colonnes supprimées: cb_person_cred_hist_length, loan_percent_income')
model_4 = executer_modeles(['cb_person_cred_hist_length', 'loan_percent_income', 'loan_amnt', 'loan_int_rate'], 'Colonnes supprimées: cb_person_cred_hist_length, loan_percent_income, loan_amnt, loan_int_rate')
tout_les_modeles = model_1 + model_2 + model_3 + model_4


# Dans le contexte des prêts, le champ "loan_status" fait référence à l'état de remboursement du prêt. Dans ce cas précis :
# 
# - 0 (non-défaut) indique que l'emprunteur est à jour avec les paiements du prêt, c'est-à-dire qu'il n'y a pas de retards ou de défauts.
# - 1 (défaut) indique que l'emprunteur ne paie pas le prêt conformément à l'accord, c'est-à-dire qu'il est en retard avec les paiements ou qu'il a déjà été considéré comme en défaut.
# 
# 
# En d'autres termes :
# 
# - Non-défaut (0) : Le prêt est remboursé régulièrement et il n'y a pas de problèmes de paiement.
# - Défaut (1) : Le prêt présente des problèmes de paiement, tels que des retards ou un manque de paiement.
# 
# Dans le contexte d'un modèle qui prédit des valeurs 0 ou 1 pour le champ "loan_status", voici ce que chaque sigle TP, TN, FP et FN représente :
# 
# TP (Vrai Positif) :
# - Prévision : 1 (défaut)
# - Réalité : 1 (défaut)
# - Signification : Le modèle a prédit correctement que le prêt serait en défaut.
# 
# 
# TN (Vrai Négatif) :
# - Prévision : 0 (non-défaut)
# - Réalité : 0 (non-défaut)
# - Signification : Le modèle a prédit correctement que le prêt ne serait pas en défaut.
# 
# 
# FP (Faux Positif) :
# - Prévision : 1 (défaut)
# - Réalité : 0 (non-défaut)
# - Signification : Le modèle a prédit incorrectement que le prêt serait en défaut, alors qu'en réalité il ne l'était pas.
# 
# <span style="font-weight: bold;">FN (Faux Négatif): (Desirable moin)</span>
# - Prévision : 0 (non-défaut)
# - Réalité : 1 (défaut)
# - Signification : Le modèle a prédit incorrectement que le prêt ne serait pas en défaut, alors qu'en réalité il l'était.
# 

# In[210]:


fn_plus_bas = min(tout_les_modeles, key=lambda x: x['FN'])
print("\033[1mModele avec FN plus bas\033[0m")
print(f'Description : {fn_plus_bas["Description"]}')
print(f'Algorithme : {fn_plus_bas["Algorithme"]}')
print(f'Faux Negatif : {fn_plus_bas["FN"]}')

plus_precis = max(tout_les_modeles, key=lambda x: x['Précision'])
print("\033[1mModele plus precis\033[0m")
print(f'Description : {plus_precis["Description"]}')
print(f'Algorithme : {plus_precis["Algorithme"]}')
print(f'Précision : {plus_precis["Précision"]}')

import joblib
joblib.dump(fn_plus_bas['Modele'], "../models/modele_fn_plus_bas.pkl")

