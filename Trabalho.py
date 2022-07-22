# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:31:15 2022

@author: Eduarda, Francisco e Rui
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

#Exercício 1

#ler ficheiro e carregar para o DataFrame ficheiro
ficheiro = pd.read_csv("C:/uni/Intro Ciencia Dados/Trabalho 1/owid-covid-data.csv", index_col=0)

#Carregar para o DataFrame americaSul apenas da América do Sul
americaSul = ficheiro.loc[ficheiro.continent == 'South America']

#Guardar o DataFrame americaSul para o ficheiro csv
americaSul.to_csv("C:/uni/Intro Ciencia Dados/Trabalho 1/america-sul.csv")

#ler ficheiro criado e carregar para o DataFrame ficheiroSul, index é a coluna data
ficheiroSul = pd.read_csv("C:/uni/Intro Ciencia Dados/Trabalho 1/america-sul.csv", index_col="date", parse_dates=True)


#Exercício 2

#Gráfico com a evolução da vacinação nos países Argentina, Brasil, Chile e Uruguai
plt.figure(figsize=(16,6))
plt.title("Evolução da vacinação nos países Argentina, Brasil, Chile e Uruguai")
plt.xlabel("Data")
plt.ylabel("Total de Vacinações")

#total de vacinações dos países ao longo dos dias
sns.lineplot(data=ficheiroSul[ficheiroSul.location == 'Argentina'].total_vaccinations, label="Argentina")
sns.lineplot(data=ficheiroSul[ficheiroSul.location == 'Brazil'].total_vaccinations, label="Brasil")
sns.lineplot(data=ficheiroSul[ficheiroSul.location == 'Chile'].total_vaccinations, label="Chile")
sns.lineplot(data=ficheiroSul[ficheiroSul.location == 'Uruguay'].total_vaccinations, label="Uruguai")

#Exercício 3
ficheiroM = pd.read_csv("C:/uni/Intro Ciencia Dados/Trabalho 1/america-sul.csv", index_col=0)

#Carregar para o DataFrame totalMortes apenas nos países pedidos, o número total de mortes do último dia e a data
totalMortes = ficheiroM.loc[(ficheiroM.location.isin(['Argentina', 'Brazil', 'Chile', 'Uruguay'])) & (ficheiroM.date == ficheiroM.date.max()), ['total_deaths','date','location']]

#Lista com o número total de mortes no último dia
x = ([int(totalMortes[totalMortes.location == 'Argentina'].total_deaths), int(totalMortes[totalMortes.location == 'Brazil'].total_deaths), 
     int(totalMortes[totalMortes.location == 'Chile'].total_deaths), int(totalMortes[totalMortes.location == 'Uruguay'].total_deaths)])

#gráfico circular com o número de mortes total no último dia do Dataset, nos países Argentina, Brasil, Chile e Uruguai
label=["Argentina", "Brasil", "Chile", "Uruguai"]
plt.title("Número de mortes total no último dia do Dataset, nos países Argentina, Brasil, Chile e Uruguai")
plt.pie(x, labels = label, autopct='%.f%%')

#Exercício 4

#função que mostra o dia em que houve mais casos positivos no país da América do Sul
def maxCasos():
    pais = ""
    
    #Verificar se é escrito o nome do país corretamente
    while ((pais != "Argentina") & (pais != "Bolivia") & (pais != "Brazil") & (pais != "Chile") & (pais != "Colombia") & (pais != "Ecuador") & 
           (pais != "Falkland Islands") & (pais != "Guyana") & (pais != "Paraguay") &  (pais != "Peru") & (pais != "Suriname") & (pais != "Uruguay") & (pais != "Venezuela")):
       
        #inserção do nome do país
        pais = input("País da América do Sul: ")
    
    #número total de casos do país escolhido pelo utilizador
    num = int(ficheiroM[ficheiroM.location == pais].new_cases.max())
    
    #dia com mais casos do país
    dia = (ficheiroM[(ficheiroM.location == pais) & (ficheiroM.new_cases == num)].date.max())
    
    return print(f"Dia com mais casos positivos no país {pais}: {dia} com {num} casos.")

maxCasos()

#Exercício 5
casos = ficheiro.new_cases
mortes = ficheiro.new_deaths

#gráfico de dispersão com regressão linear com o Relacionamento entre o número de mortes diários e novos casos positivos de Covid diários
plt.figure(figsize=(16,6))
plt.title("Relacionamento entre o número de mortes diários e novos casos positivos de Covid diários")
x, y = pd.Series(casos, name="Casos Diários Positivos"), pd.Series(mortes, name="Mortes Diárias")
sns.regplot(x=x, y=y)

#Exercício 6
ficheiroD = pd.read_csv("C:/uni/Intro Ciencia Dados/Trabalho 1/owid-covid-data.csv", index_col="date", parse_dates=True)
casosP = ficheiroD.new_cases

#gráfico com a evolução de casos positivos de Covid diários
plt.figure(figsize=(16,6))
plt.title("Evolução de casos positivos de Covid diários")
plt.xlabel("Data")
plt.ylabel("Casos Diários Positivos")
sns.lineplot(data=casosP)

#estatística descritiva dos novos casos
casosP.describe().apply("{0:.0f}".format)


df = pd.read_csv("C:/uni/Intro Ciencia Dados/Trabalho 1/owid-covid-data.csv")

#dataframe com as colunas data e novos casos
df = df.loc[:,['date','new_cases']]
df.index = pd.to_datetime(df['date'], format='%Y-%m-%d')
del df['date']

#substituir valores infinitos por 0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

#gráfico com a evolução de casos positivos de Covid diários com os valores corrigidos
sns.set()
plt.figure(figsize=(16,6))
plt.title("Evolução de casos positivos de Covid diários")
plt.ylabel('Casos Diários Positivos')
plt.xlabel('Data')
plt.xticks(rotation=45)
plt.plot(df.index, df['new_cases'], )

#dataframes para usar e verificar os dados para a previsão
train = df[df.index < pd.to_datetime("2020-07-01", format='%Y-%m-%d')]
test = df[df.index > pd.to_datetime("2020-07-01", format='%Y-%m-%d')]

#gráfico com os dataframes train e test
plt.figure(figsize=(16,6))
plt.title("Evolução de casos positivos de Covid diários")
plt.ylabel('Casos Diários Positivos')
plt.xlabel('Data')
plt.xticks(rotation=45)
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.show()

#modelo auto regressivo para prever o número de casos
from statsmodels.tsa.statespace.sarimax import SARIMAX
y = train['new_cases']

ARMAmodel = SARIMAX(y, order = (1, 2, 1))
ARMAmodel = ARMAmodel.fit()
y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 

#ver o erro entre o número de casos e a previsão
from sklearn.metrics import mean_squared_error
arma_rmse = np.sqrt(mean_squared_error(test["new_cases"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)

#gráfico com os dataframes test, train e a previsão
plt.figure(figsize=(16,6))
plt.title("Evolução de casos positivos de Covid diários")
plt.ylabel('Casos Diários Positivos')
plt.xlabel('Data')
plt.xticks(rotation=45)
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
plt.legend()
plt.show()


import datetime
import random
df = df[['new_cases']]
df.dropna(inplace=True)
last_close = df['new_cases'][-1]
last_date = df.iloc[-1].name.timestamp()
df['Predictions'] = np.nan

for i in range(1000):
    
    modifier = random.randint(-100, 105) / 10000 + 1
    last_close *= modifier
    next_date = datetime.datetime.fromtimestamp(last_date)
    last_date += 10000

    df.loc[next_date] = [np.nan, last_close]

#gráfico com os novos casos e a previsão
plt.figure(figsize=(16,6))
plt.title("Evolução de casos positivos de Covid diários")
plt.ylabel('Casos Diários Positivos')
plt.xlabel('Data')
plt.xticks(rotation=45)
#df['new_cases'].plot()
df['Predictions'].plot()
plt.legend()




