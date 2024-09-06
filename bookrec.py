import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from mlxtend.frequent_patterns import association_rules, fpgrowth
from datetime import datetime
from PIL import Image
from mlxtend.preprocessing import TransactionEncoder

#image= Image.open('peminjaman.jpg')

##Judul
st.write(""" 
# Web App Recommendation
         
Website Aplikasi Rekomendasi Buku Perpustakaan UNP
         
***
""")
#Input Data
dataset= pd.read_excel('DATA PENELITIAN4.xlsx')
dataset= dataset[['Transaksi', 'Judul', 'Tahun Masuk', 'Fakultas', 'Hari']]
dataset.columns= ['Transaksi', 'Judul', 'Tahun_Masuk', 'Fakultas', 'Hari']
dataset['Tahun_Masuk']= dataset['Tahun_Masuk'].astype(str)

#Analsis Deskriptif Data Data 
##1
st.subheader('Explorasi Data')
fig, ax = plt.subplots()
dataset['Hari'].value_counts().plot(kind='bar', color='blue')
ax.set_xlabel('Hari')
ax.set_ylabel('Jumlah')
ax.set_title('Jumlah Peminjaman Buku menurut Hari')
st.pyplot(fig)

##2
fig1, ax1 = plt.subplots()
dataset['Fakultas'].value_counts().plot(kind= 'bar', color= 'red')
ax1.set_xlabel('Fakultas')
ax1.set_ylabel('Jumlah')
ax1.set_title('Jumlah Peminjaman menurut Fakultas')
st.pyplot(fig1)

##definisikan data

def get_data(Tahun_masuk= '', Fakultas= '', Hari=''):
    dataset= dataset.copy()
    filter_data= dataset.loc[
        (dataset["Tahun_masuk"].str.contains(Tahun_masuk))&
        (dataset['Fakultas'].str.contains(Fakultas))&
        (dataset["Hari"].str.contains(Hari))
    ]
    return filter_data if filter_data.shape[0] else "tidak ada hasil"

##membuat selectbox
isi= pd.read_excel('JUDUL BUKU.xlsx')
isi_item= isi['Judul'].values.tolist()

##transformasi data
Judul= dataset[['Transaksi','Judul']] 
transactions = Judul.groupby('Transaksi')['Judul'].apply(list).tolist()
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
te_columns = te.columns_
dataset_encoded = pd.DataFrame(te_array, columns=te_columns)


##menjalankan algoritma FP-Growth
support_threshold= 0.0009
freq_items= fpgrowth(dataset_encoded, min_support=support_threshold, use_colnames= True)

##menampilkan 10 item paling sering dipinjam
st.subheader("10 Judul Paling Pering Dipinjam")
freq_items["itemsets"][1:10]

def User_input_features():
    Item= st.selectbox("Judul", isi_item)
    Tahun_Masuk= st.selectbox("Tahun Masuk", ["2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022","2023"])
    Fakultas= st.selectbox("Fakultas", ["FIP", "FBS", "FMIPA", "FIS", "FT", "FIK", "FPP", "FPS", "OTHERS"])
    Hari= st.select_slider("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
    return Item, Tahun_Masuk, Fakultas, Hari 

Item, Tahun_Masuk, Fakultas, Hari= User_input_features()

##Association Rules
metric= "lift"
min_threshold= 1
rule= association_rules(freq_items, metric=metric, min_threshold=min_threshold, )[["antecedents", "consequents", "support", "confidence", "lift"]]
rule.sort_values('confidence', ascending= False, inplace= True)

def parse_list(x):
    if isinstance(x, frozenset):
        return ", ".join(map(str, x))
    else:
        return str(x)
    
def return_item_judul(item_antecedents):
    dataset= rule[["antecedents", "consequents"]].copy()
    dataset["antecedents"]= dataset["antecedents"].apply(parse_list)
    dataset["consequents"]= dataset["consequents"].apply(parse_list)
    
    if item_antecedents in dataset["antecedents"].values:
        return list(dataset.loc[dataset["antecedents"] == item_antecedents].iloc[0, :])
    else:
        return None

if type(rule) != type("tidak ada hasil"):
    st.markdown("Rekomendasi Buku Perpustakaan: ")
    result = return_item_judul(Item)
    if result:
        st.success(f"Jika meminjam **{Item}**, maka dapat meminjam **{result[1]}** di perpustakaan")
    else:
        st.error(f"Tidak ada rekomendasi untuk **{Item}**")
