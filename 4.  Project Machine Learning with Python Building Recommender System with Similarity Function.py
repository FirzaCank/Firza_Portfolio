#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import library yang dibutuhkan
import pandas as pd
import numpy as np

#lakukan pembacaan dataset
movie_rating_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/movie_rating_df.csv') #untuk menyimpan movie_rating_df.csv

#tampilkan 5 baris teratas dari movive_rating_df
print(movie_rating_df.head())

#tampilkan info mengenai tipe data dari tiap kolom
print(movie_rating_df.info())


# In[16]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

director_writers = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/directors_writers.csv')
print(director_writers)


# In[17]:


#Mengubah director_name menjadi list
director_writers['director_name'] = director_writers['director_name'].apply(lambda row: row.split(',')) # sama aja  row diganti x
director_writers['writer_name'] = director_writers['writer_name'].apply(lambda row: row.split(',')) # apply(lambda x: x.split(','))

#Tampilkan 5 data teratas
print(director_writers.head())


# In[30]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

name_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/actor_name.csv')
print(name_df)


# In[31]:


name_df = name_df[['nconst','primaryName','knownForTitles']]

#Tampilkan 5 baris teratas dari name_df
print(name_df.head())


# In[32]:


#Melakukan pengecekan variasi
print(name_df['knownForTitles'].apply(lambda x: len(x.split(','))).unique())


# In[33]:


#Mengubah knownForTitles menjadi list of list
name_df['knownForTitles'] = name_df['knownForTitles'].apply(lambda x: x.split(','))

#Mencetak 5 baris teratas
print(name_df.head())


# In[61]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

name_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/actor_name.csv')
name_df = name_df[['nconst','primaryName','knownForTitles']]
name_df['knownForTitles'] = name_df['knownForTitles'].apply(lambda x: x.split(','))

#menyiapkan bucket untuk dataframe
df_uni = []

for x in ['knownForTitles']:
    #mengulang index dari tiap baris sampai tiap elemen dari knownForTitles
    idx = name_df.index.repeat(name_df['knownForTitles'].str.len())
   
   #memecah values dari list di setiap baris dan menggabungkan nya dengan rows lain menjadi dataframe
    df1 = pd.DataFrame({
        x: np.concatenate(name_df[x].values)
    })
    
    #mengganti index dataframe tersebut dengan idx yang sudah kita define di awal
    df1.index = idx
    #untuk setiap dataframe yang terbentuk, kita append ke dataframe bucket
    df_uni.append(df1)
    
#menggabungkan semua dataframe menjadi satu (karena loop)
df_concat = pd.concat(df_uni, axis=1)

#left join dengan value dari dataframe yang awal
#Gabungan dari name_df yang Udh di drop kolom knwonfortitles,dengan df_concat(hasil loop korespondensi 1-1)
unnested_df = df_concat.join(name_df.drop(['knownForTitles'], 1), how='left')

#select kolom sesuai dengan dataframe awal
unnested_df = unnested_df[name_df.columns.tolist()]
print(unnested_df)


# In[63]:


unnested_drop = unnested_df.drop(['nconst'], axis=1)
print(unnested_drop)


# In[68]:


for col in ['primaryName']:

#agregasi kolom PrimaryName sesuai group_col yang sudah di define di atas
    dfi = unnested_drop.groupby(['knownForTitles'])[col].apply(list)
dfi


# In[69]:


df_uni = []

for col in ['primaryName']:

#agregasi kolom PrimaryName sesuai group_col yang sudah di define di atas
    dfi = unnested_drop.groupby(['knownForTitles'])[col].apply(list)
    df_uni.append(dfi)
df_uni


# In[71]:


df_grouped = pd.concat(df_uni, axis=1).reset_index()
print(df_grouped)


# In[72]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

name_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/actor_name.csv')
name_df = name_df[['nconst','primaryName','knownForTitles']]
name_df['knownForTitles'] = name_df['knownForTitles'].apply(lambda x: x.split(','))

df_uni = []
for x in ['knownForTitles']:
    idx = name_df.index.repeat(name_df['knownForTitles'].str.len())
    df1 = pd.DataFrame({
        x: np.concatenate(name_df[x].values)
    })
    df1.index = idx
    df_uni.append(df1)

df_concat = pd.concat(df_uni, axis=1)
unnested_df = df_concat.join(name_df.drop(['knownForTitles'], 1), how='left')
unnested_df = unnested_df[name_df.columns.tolist()]

unnested_drop = unnested_df.drop(['nconst'], axis=1)

#menyiapkan bucket untuk dataframe
df_uni = []

for col in ['primaryName']:
    #agregasi kolom PrimaryName sesuai group_col yang sudah di define di atas
    dfi = unnested_drop.groupby(['knownForTitles'])[col].apply(list)
    #Lakukan append
    df_uni.append(dfi)
	
#Menggabungkan semua dataframe menjadi satu
df_grouped = pd.concat(df_uni, axis=1).reset_index()
df_grouped.columns = ['knownForTitles','cast_name'] #mengubah nama kolom
print(df_grouped) #tabel cast/ cast table 


# In[79]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

movie_rating_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/movie_rating_df.csv') #movie_rating

director_writers = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/directors_writers.csv') #director_writers
director_writers['director_name'] = director_writers['director_name'].apply(lambda row: row.split(','))
director_writers['writer_name'] = director_writers['writer_name'].apply(lambda row: row.split(','))

name_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/actor_name.csv') #nama-nama
name_df = name_df[['nconst','primaryName','knownForTitles']]
name_df['knownForTitles'] = name_df['knownForTitles'].apply(lambda x: x.split(','))

df_uni = []
for x in ['knownForTitles']:
    idx = name_df.index.repeat(name_df['knownForTitles'].str.len())
    df1 = pd.DataFrame({
        x: np.concatenate(name_df[x].values)
    })
    df1.index = idx
    df_uni.append(df1)

df_concat = pd.concat(df_uni, axis=1)
unnested_df = df_concat.join(name_df.drop(['knownForTitles'], 1), how='left')
unnested_df = unnested_df[name_df.columns.tolist()]

unnested_drop = unnested_df.drop(['nconst'], axis=1)
df_uni = []
for col in ['primaryName']:
    dfi = unnested_drop.groupby(['knownForTitles'])[col].apply(list)
    df_uni.append(dfi)
df_grouped = pd.concat(df_uni, axis=1).reset_index()
df_grouped.columns = ['knownForTitles','cast_name']

#join antara movie table dan cast table 
base_df = pd.merge(df_grouped, movie_rating_df, left_on='knownForTitles', right_on='tconst', how='inner')

#join antara base_df dengan director_writer table
base_df = pd.merge(base_df, director_writers, left_on='tconst', right_on='tconst', how='left')
print(base_df.head())


# In[83]:


#Melakukan drop terhadap kolom knownForTitles
base_drop = base_df.drop(['knownForTitles'], axis=1)
print(base_drop.info())

#Mengganti nilai NULL pada kolom genres dengan 'Unknown'
base_drop['genres'] = base_drop['genres'].fillna('unknown')

#Melakukan perhitungan jumlah nilai NULL pada tiap kolom
print(base_drop.isnull().sum())

#Mengganti nilai NULL pada kolom dorector_name dan writer_name dengan 'Unknown'
base_drop[['director_name','writer_name']] = base_drop[['director_name','writer_name']].fillna('unknown')

#karena value kolom genres terdapat multiple values, jadi kita akan bungkus menjadi list of list
base_drop['genres'] = base_drop['genres'].apply(lambda x: x.split(','))


# In[88]:


#Drop kolom tconst, isAdult, endYear, originalTitle
base_drop2 = base_drop.drop(['tconst','isAdult','endYear','originalTitle'], axis=1)

base_drop2 = base_drop2[['primaryTitle','titleType','startYear','runtimeMinutes','genres','averageRating','numVotes','cast_name','director_name','writer_name']]

# Gunakan petunjuk!
base_drop2.columns = ['title','type','start','duration','genres','rating','votes','cast_name','director_name','writer_name']
print(base_drop2.head())


# In[89]:


#Klasifikasi berdasar title, cast_name, genres, director_name, dan writer_name
feature_df = base_drop2[['title','cast_name','genres','director_name','writer_name']]

#Tampilkan 5 baris teratas
print(feature_df.head())


# In[93]:


def sanitize(x):
    try: #try untuk tes code
        #kalau cell berisi list
        if isinstance(x, list): #apakah x adalah list?
            return [i.replace(' ','').lower() for i in x]
        #kalau cell berisi string
        else: #apabila x bukan list
            return [x.replace(' ','').lower()]
    except: #apabila error, except akan handle print(x)
        print(x)
        
#Kolom : cast_name, genres, writer_name, director_name        
feature_cols = ['cast_name','genres','writer_name','director_name']

#Apply function sanitize 
for col in feature_cols:
    feature_df[col] = feature_df[col].apply(sanitize)

print(feature_df.head())


# In[100]:


#kolom yang digunakan : cast_name, genres, director_name, writer_name
def soup_feature(x):
    return ' '.join(x['cast_name']) + ' ' + ' '.join(x['genres']) + ' ' + ' '.join(x['director_name']) + ' ' + ' '.join(x['writer_name'])

#membuat soup menjadi 1 kolom 
feature_df['soup'] = feature_df.apply(soup_feature, axis=1)

feature_df['soup']


# In[116]:


#import CountVectorizer 
from sklearn.feature_extraction.text import CountVectorizer

#definisikan CountVectorizer dan mengubah soup tadi menjadi bentuk vector
count = CountVectorizer(stop_words= 'english')
count_matrix = count.fit_transform(feature_df['soup'])

print(count)
print(count_matrix.shape)
print(count_matrix)


# In[102]:


#Import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

#Gunakan cosine_similarity antara count_matrix 
cosine_sim = cosine_similarity(count_matrix, count_matrix)

#print hasilnya
print(cosine_sim)


# In[104]:


print(feature_df)


# In[106]:


indices = pd.Series(feature_df.index, index=feature_df['title']).drop_duplicates()

def content_recommender(title):
    #mendapatkan index dari judul film (title) yang disebutkan
    idx = indices[title]

    #menjadikan list dari array similarity cosine sim 
    #hint: cosine_sim[idx]
    sim_scores = list(enumerate(cosine_sim[idx]))

    #mengurutkan film dari similarity tertinggi ke terendah
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)

    #untuk mendapatkan list judul dari item kedua sampe ke 11
    sim_scores = sim_scores[1:11]

    #mendapatkan index dari judul-judul yang muncul di sim_scores
    movie_indices = [i[0] for i in sim_scores]

    #dengan menggunakan iloc, kita bisa panggil balik berdasarkan index dari movie_indices
    return base_df.iloc[movie_indices]

#aplikasikan function di atas
print(content_recommender('The Lion King'))


# In[107]:


indices = pd.Series(feature_df.index, index=feature_df['title']).drop_duplicates()
print(indices)


# In[110]:


#mendapatkan index dari judul film (title) yang disebutkan
idx = indices['The Lion King']
idx


# In[111]:


sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores


# In[112]:


sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
sim_scores    


# In[113]:


sim_scores = sim_scores[1:11]
sim_scores


# In[114]:


#mendapatkan index dari judul-judul yang muncul di sim_scores
movie_indices = [i[0] for i in sim_scores]
movie_indices    


# In[ ]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

movie_rating_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/movie_rating_df.csv')

director_writers = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/directors_writers.csv')
director_writers['director_name'] = director_writers['director_name'].apply(lambda row: row.split(','))
director_writers['writer_name'] = director_writers['writer_name'].apply(lambda row: row.split(','))

name_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/actor_name.csv')
name_df = name_df[['nconst','primaryName','knownForTitles']]
name_df['knownForTitles'] = name_df['knownForTitles'].apply(lambda x: x.split(','))

df_uni = []
for x in ['knownForTitles']:
    idx = name_df.index.repeat(name_df['knownForTitles'].str.len())
    df1 = pd.DataFrame({
        x: np.concatenate(name_df[x].values)
    })
    df1.index = idx
    df_uni.append(df1)

df_concat = pd.concat(df_uni, axis=1)
unnested_df = df_concat.join(name_df.drop(['knownForTitles'], 1), how='left')
unnested_df = unnested_df[name_df.columns.tolist()]

unnested_drop = unnested_df.drop(['nconst'], axis=1)
df_uni = []
for col in ['primaryName']:
    dfi = unnested_drop.groupby(['knownForTitles'])[col].apply(list)
    df_uni.append(dfi)
df_grouped = pd.concat(df_uni, axis=1).reset_index()
df_grouped.columns = ['knownForTitles','cast_name']

base_df = pd.merge(df_grouped, movie_rating_df, left_on='knownForTitles', right_on='tconst', how='inner')
base_df = pd.merge(base_df, director_writers, left_on='tconst', right_on='tconst', how='left')

base_drop = base_df.drop(['knownForTitles'], axis=1)
base_drop['genres'] = base_drop['genres'].fillna('Unknown')
base_drop[['director_name','writer_name']] = base_drop[['director_name','writer_name']].fillna('unknown')
base_drop['genres'] = base_drop['genres'].apply(lambda x: x.split(','))

base_drop2 = base_drop.drop(['tconst','isAdult','endYear','originalTitle'], axis=1)
base_drop2 = base_drop2[['primaryTitle','titleType','startYear','runtimeMinutes','genres','averageRating','numVotes','cast_name','director_name','writer_name']]
base_drop2.columns = ['title','type','start','duration','genres','rating','votes','cast_name','director_name','writer_name']

feature_df = base_drop2[['title','cast_name','genres','director_name','writer_name']]

def sanitize(x):
    try:
        if isinstance(x, list):
            return [i.replace(' ','').lower() for i in x]
        else:
            return [x.replace(' ','').lower()]
    except:
        print(x)
feature_cols = ['cast_name','genres','writer_name','director_name']

for col in feature_cols:
    feature_df[col] = feature_df[col].apply(sanitize)

def soup_feature(x):
    return ' '.join(x['cast_name']) + ' ' + ' '.join(x['genres']) + ' ' + ' '.join(x['director_name']) + ' ' + ' '.join(x['writer_name'])

feature_df['soup'] = feature_df.apply(soup_feature, axis=1)

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(feature_df['soup'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(feature_df.index, index=feature_df['title']).drop_duplicates()

def content_recommender(title):
    #mendapatkan index dari judul film (title) yang disebutkan
    idx = indices[title]

    #menjadikan list dari array similarity cosine sim 
    #hint: cosine_sim[idx]
    sim_scores = list(enumerate(cosine_sim[idx]))

    #mengurutkan film dari similarity tertinggi ke terendah
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)

    #untuk mendapatkan list judul dari item kedua sampe ke 11
    sim_scores = sim_scores[1:11]

    #mendapatkan index dari judul-judul yang muncul di sim_scores
    movie_indices = [i[0] for i in sim_scores]

    #dengan menggunakan iloc, kita bisa panggil balik berdasarkan index dari movie_indices
    return base_df.iloc[movie_indices]

#aplikasikan function di atas
print(content_recommender('The Lion King'))


# In[ ]:




