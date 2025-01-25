import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import traceback
from decimal import Decimal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from psycopg2 import OperationalError


from Backend.Database.helpers import get_db_connection, get_non_numeric_cols, get_block_indexes
from Backend.Util import TableManager, Table
from Configuration.config import Config

from tqdm import tqdm
import json
from datetime import datetime
import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense


from gensim.models import Word2Vec, FastText
from datasketch import MinHash, MinHashLSH
import pickle as cPickle

tokenizer = None
model = None
scaler = None
max_len = None
embedding_dim = 8
base_model_file_dir = "./SavedFiles/Models/fasttexts/"


class TablePCA:
    def __init__(self, name, pca_df, eig_values_='', eig_vectors_=''):
        self.table_name = name
        self.pca_df = pca_df
        self.eig_values = eig_values_
        self.eig_vectors = eig_vectors_

def get_column_dtype(table_name, cursor):
    sql ='''
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{tb}';
    '''
    loop_count = 0
    conn = get_db_connection()
    cursor = conn.cursor()
    while (True):
        try:
            cursor.execute(sql.format(tb=table_name))
            df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            if Config.use_locking and Config.lock_requested:
                os.remove(Config.lock_file)
                os.remove(Config.lock_request_file)
                Config.lock_acquired = False
                Config.lock_requested = False
                print("Releasing the lock.")
        except OperationalError as e:
            loop_count += 1
            conn = get_db_connection()
            cursor = conn.cursor()
            continue
        # print('Exiting the loop')
        break


    res = {}
    for idx, row in df.iterrows():
        res[row['column_name']] = row['data_type']
    return res


def generate_string_encoder(vals, embedding_dim):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(vals)

    sequences = tokenizer.texts_to_sequences(vals)

    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    scaler = MinMaxScaler()
    padded_sequences = scaler.fit_transform(padded_sequences)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(embedding_dim, activation='relu'))  # Encoder
    model.add(Dense(max_len, activation='sigmoid')) 

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(padded_sequences, padded_sequences, epochs=25)
    encoder_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    return tokenizer, encoder_model, scaler, max_len


def smart_get_table_data(table_name, removing_features, cursor, enc_model='w'):
    global embedding_dim
    if enc_model == 'w':
        embedding_dim = 8 #32
    if enc_model == 'c':
        embedding_dim = 32
    sql = 'select * from ' + table_name
    if Config.is_test:
        sql = sql + ' limit 1000;'
    s = time.time()

    loop_count = 0
    while (True):
        try:
            cursor.execute(sql)
            df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            cols = [desc[0] for desc in cursor.description]
            if Config.use_locking and Config.lock_requested:
                os.remove(Config.lock_file)
                os.remove(Config.lock_request_file)
                Config.lock_acquired = False
                Config.lock_requested = False
                print("Releasing the lock.")
        except OperationalError as e:
            loop_count += 1
            conn = get_db_connection()
            cursor = conn.cursor()
            continue
        # print('Exiting the loop')
        break
    
    col_dtype_dict = get_column_dtype(table_name, cursor)
    all_str_vals = []
    str_cols = []
    print('des')

    total_rows = len(df)
    for feature in cols:
        if Config.do_string_encoding:
            if col_dtype_dict[feature] in ['character varying', 'character', 'text']:
                if 'tpch' in Config.db_name and ('address' in feature or 'comment' in feature):
                    print(f'Passing column {feature}!')
                    df = df.drop(feature, axis=1)
                    continue

                str_cols.append(feature)
            elif 'date' in col_dtype_dict[feature] or 'timestamp' in col_dtype_dict[feature]:
                df[feature] = df[feature].apply(lambda x: datetime.timestamp(pd.to_datetime(x)) if pd.notnull(x) else 0)
            elif  'time' in col_dtype_dict[feature]:
                default_date = datetime(1970, 1, 1) 
                df[feature] = df[feature].apply(lambda x: datetime.combine(default_date, x))
                df[feature] = df[feature].astype(int) // 10**9  

        else:
            df = df.drop(feature, axis=1)
    
    if len(str_cols) == 0:
        return df

    e = time.time()
    print(f'Loaded and checked table {table_name} in {e-s}s;\tstr_cols are: {str_cols}')
    
    s = time.time()
    cumol_size = 0
    for col in str_cols:
        df[col] = df[col].fillna('-far-')
        col_vals = df[col].str.lower().to_list()
        if enc_model == 'c':
            all_str_vals = all_str_vals + list(set(col_vals))
        if enc_model == 'w':
            all_str_vals.append(list(set(col_vals)))
        cumol_size += len(all_str_vals[-1])
    e = time.time()
    print(f'\tProccessed columns in {e-s}s, got {len(all_str_vals)} unique strings.')

    
    global tokenizer, model, scaler, max_len, base_model_file_dir
    if enc_model == 'w':
        if 'NewMLPft' in Config.config_suffix:
            all_str_vals = [item for sublist in all_str_vals for item in sublist]
            s = time.time()
            model = model = FastText(all_str_vals, vector_size=embedding_dim, window=3, min_count=1, epochs=10)
            e = time.time()
            print(f'\tCreated and trained fasttext model in {e-s}s.')
            model.save(f"{base_model_file_dir}ftxt_{Config.db_name}_{table_name}_{col}_{Config.config_suffix}.model")
            e1 = time.time()
            print(f'\tStored the model in {e1-e}s.')
        
        elif 'NewMLPmh' not in Config.config_suffix:
            s = time.time()
            model = Word2Vec(all_str_vals, vector_size=embedding_dim, min_count=1)
            model.build_vocab(all_str_vals)
            model.train(all_str_vals, total_examples=len(all_str_vals), epochs=10, report_delay=1)
            e = time.time()
            print(f'\tCreated and trained word2vec model in {e-s}s.')
    
    if enc_model == 'c':
        tokenizer, model, scaler, max_len = generate_string_encoder(list(set(all_str_vals)), embedding_dim)

    print(len(df.columns.to_list()))
    ss = []
    for col in str_cols:
        s = time.time()
        ss.append(s)
        if enc_model == 'w':
            if 'NewMLPmh' not in Config.config_suffix:
                df[col] = df[col].apply(get_embedding)
            else:
                df[col] = df[col].apply(get_embeddingmh)
        if enc_model == 'c':
            df[col] = df[col].apply(get_embedding2)
        e = time.time()
        df2 = pd.DataFrame(df[col].to_list(), columns=[f'{col}_{i}' for i in range(embedding_dim)])
        df = pd.concat([df, df2], axis=1)
        df = df.drop(col, axis=1)
        e1 = time.time()
        print(f'\tEncoded {col} in {e-s}s, updated df in {e1-e}')

    print(len(df.columns.to_list()))

    return df


def smart_get_table_data_ftune_fasttext(table_name):
    global embedding_dim
    embedding_dim = 8 #32
    conn = get_db_connection(dbname=Config.ftune_base_db_name)
    cursor = conn.cursor()
    sql = 'select * from ' + table_name
    if Config.is_test:
        sql = sql + ' limit 1000;'
    s = time.time()

    loop_count = 0
    while (True):
        try:
            cursor.execute(sql)
            base_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            if Config.use_locking and Config.lock_requested:
                os.remove(Config.lock_file)
                os.remove(Config.lock_request_file)
                Config.lock_acquired = False
                Config.lock_requested = False
                print("Releasing the lock.")
        except OperationalError as e:
            loop_count += 1
            conn = get_db_connection(dbname=Config.ftune_base_db_name)
            cursor = conn.cursor()
            continue
        # print('Exiting the loop')
        break

    conn2 = get_db_connection(dbname=Config.ftune_target_db_name)
    cursor2 = conn2.cursor()
    loop_count = 0
    while (True):
        try:
            cursor2.execute(sql)
            df = pd.DataFrame(cursor2.fetchall(), columns=[desc[0] for desc in cursor2.description])
            if Config.use_locking and Config.lock_requested:
                os.remove(Config.lock_file)
                os.remove(Config.lock_request_file)
                Config.lock_acquired = False
                Config.lock_requested = False
                print("Releasing the lock.")
        except OperationalError as e:
            loop_count += 1
            conn2 = get_db_connection(dbname=Config.ftune_target_db_name)
            cursor2 = conn2.cursor()
            continue
        # print('Exiting the loop')
        break
    

    cols = [desc[0] for desc in cursor.description]
    col_dtype_dict = get_column_dtype(table_name, cursor)
    all_str_vals = []
    str_cols = []
    print('des')

    total_rows = len(base_df)
    for feature in cols:
        if Config.do_string_encoding:
            if col_dtype_dict[feature] in ['character varying', 'character', 'text']:

                if 'tpch' in Config.db_name and ('address' in feature or 'comment' in feature):
                    print(f'Passing column {feature}!')
                    df = df.drop(feature, axis=1)
                    base_df = base_df.drop(feature, axis=1)
                    continue

                str_cols.append(feature)
            elif 'date' in col_dtype_dict[feature] or 'timestamp' in col_dtype_dict[feature]:
                df[feature] = df[feature].apply(lambda x: datetime.timestamp(pd.to_datetime(x)) if pd.notnull(x) else 0)
                base_df[feature] = base_df[feature].apply(lambda x: datetime.timestamp(pd.to_datetime(x)) if pd.notnull(x) else 0)
            elif  'time' in col_dtype_dict[feature]:
                default_date = datetime(1970, 1, 1) 
                df[feature] = df[feature].apply(lambda x: datetime.combine(default_date, x))
                df[feature] = df[feature].astype(int) // 10**9  

                base_df[feature] = base_df[feature].apply(lambda x: datetime.combine(default_date, x))
                base_df[feature] = base_df[feature].astype(int) // 10**9  
        else:
            df = df.drop(feature, axis=1)
            base_df = base_df.drop(feature, axis=1)
    
    if len(str_cols) == 0:
        return base_df, df

    e = time.time()
    print(f'Loaded and checked table {table_name} in {e-s}s;\tstr_cols are: {str_cols}')
    
    for col in str_cols:
        df[col] = df[col].fillna('-far-')
        base_df[col] = base_df[col].fillna('-far-')

    
    global tokenizer, model, scaler, max_len, base_model_file_dir
    s = time.time()
    cf = Config.config_suffix
    ftune_idx = cf.index('_ftune')
    cf = cf[:ftune_idx]
    model = FastText.load(f"{base_model_file_dir}ftxt_{Config.db_name}_{table_name}_{col}_{cf}.model")
    e = time.time()
    print(f'\tLoaded the model in {e-s}s.')
        
    
    for col in str_cols:
        s = time.time()
        base_df[col] = base_df[col].apply(get_embedding)
        e = time.time()
        df2 = pd.DataFrame(base_df[col].to_list(), columns=[f'{col}_{i}' for i in range(embedding_dim)])
        base_df = pd.concat([base_df, df2], axis=1)
        base_df = base_df.drop(col, axis=1)
        e1 = time.time()
        print(f'\tEncoded {col} in {e-s}s, updated base df in {e1-e}')
    
    for col in str_cols:
        s = time.time()
        df[col] = df[col].apply(get_embedding)
        e = time.time()
        df2 = pd.DataFrame(df[col].to_list(), columns=[f'{col}_{i}' for i in range(embedding_dim)])
        df = pd.concat([df, df2], axis=1)
        df = df.drop(col, axis=1)
        e1 = time.time()
        print(f'\tEncoded {col} in {e-s}s, updated target df in {e1-e}')   

    return base_df, df


def get_table_data(table_name, removing_features, cursor, enc_model='w'):
    if enc_model == 'w':
        embedding_dim = 8 #32
    if enc_model == 'c':
        embedding_dim = 32
    sql = 'select * from ' + table_name
    if Config.is_test:
        sql = sql + ' limit 1000;'

    loop_count = 0
    while (True):
        try:
            cursor.execute(sql)
            df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            if Config.use_locking and Config.lock_requested:
                os.remove(Config.lock_file)
                os.remove(Config.lock_request_file)
                Config.lock_acquired = False
                Config.lock_requested = False
                print("Releasing the lock.")
        except OperationalError as e:
            loop_count += 1
            conn = get_db_connection()
            cursor = conn.cursor()
            continue
        # print('Exiting the loop')
        break
    
    col_dtype_dict = get_column_dtype(table_name, cursor)
    all_str_vals = []
    str_cols = []
    if removing_features:
        for feature in removing_features:
            if Config.do_string_encoding:
                if col_dtype_dict[feature] == 'character varying':
                    str_cols.append(feature)
                else:
                    df = df.drop(feature, axis=1)
            else:
                df = df.drop(feature, axis=1)
    
    if len(str_cols) == 0:
        print(f'No string to encode in {table_name}')
        return df
    
    total_rows = len(df)
    for col in str_cols:
        df[col] = df[col].fillna('-far-')

        if 'tpch' in Config.db_name and 'address' in feature or 'comment' in feature:
            print(f'Passing column {feature}!')
            df = df.drop(feature, axis=1)
            continue

        col_vals = df[col].str.lower().to_list()
        if enc_model == 'c':
            all_str_vals = all_str_vals + list(set(col_vals))
        if enc_model == 'w':
            all_str_vals.append(list(set(col_vals)))
    
    global tokenizer, model, scaler, max_len
    if enc_model == 'w':
        if 'NewMLPft' in Config.config_suffix:
            print('ft')
            all_str_vals = [item for sublist in all_str_vals for item in sublist]
            s = time.time()
            model = model = FastText(all_str_vals, vector_size=embedding_dim, window=3, min_count=1, epochs=10)
            e = time.time()
            print(f'\tCreated and trained fasttext model in {e-s}s.')
            model.save(f"{base_model_file_dir}ftxt_{Config.db_name}_{table_name}_{col}_{Config.config_suffix}.model")
            e1 = time.time()
            print(f'\tStored the model in {e1-e}s.')

        elif 'NewMLPmh' not in Config.config_suffix:
            s = time.time()
            model = Word2Vec(all_str_vals, vector_size=embedding_dim, min_count=1)
            model.build_vocab(all_str_vals)
            model.train(all_str_vals, total_examples=len(all_str_vals), epochs=10, report_delay=1)
            e = time.time()
            print(f'\tCreated and trained word2vec model in {e-s}s.')
    
    if enc_model == 'c':
        tokenizer, model, scaler, max_len = generate_string_encoder(list(set(all_str_vals)), embedding_dim)

    print(len(df.columns.to_list()))
    for col in str_cols:
        if enc_model == 'w':
            if 'NewMLPmh' not in Config.config_suffix:
                df[col] = df[col].apply(get_embedding)
            else:
                df[col] = df[col].apply(get_embeddingmh)
        if enc_model == 'c':
            df[col] = df[col].apply(get_embedding2)

        df2 = pd.DataFrame(df[col].to_list(), columns=[f'{col}_{i}' for i in range(embedding_dim)])
        df = pd.concat([df, df2], axis=1)
        df = df.drop(col, axis=1)
    print(len(df.columns.to_list()))

    return df

def get_table_data_ftune_fasttext(table_name, removing_features):
    global embedding_dim
    embedding_dim = 8 #32
    conn = get_db_connection(dbname=Config.ftune_base_db_name)
    cursor = conn.cursor()
    sql = 'select * from ' + table_name
    if Config.is_test:
        sql = sql + ' limit 1000;'
    s = time.time()

    loop_count = 0
    while (True):
        try:
            cursor.execute(sql)
            base_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            if Config.use_locking and Config.lock_requested:
                os.remove(Config.lock_file)
                os.remove(Config.lock_request_file)
                Config.lock_acquired = False
                Config.lock_requested = False
                print("Releasing the lock.")
        except OperationalError as e:
            loop_count += 1
            conn = get_db_connection(dbname=Config.ftune_base_db_name)
            cursor = conn.cursor()
            continue
        # print('Exiting the loop')
        break

    conn2 = get_db_connection(dbname=Config.ftune_target_db_name)
    cursor2 = conn2.cursor()
    loop_count = 0
    while (True):
        try:
            cursor2.execute(sql)
            df = pd.DataFrame(cursor2.fetchall(), columns=[desc[0] for desc in cursor2.description])
            if Config.use_locking and Config.lock_requested:
                os.remove(Config.lock_file)
                os.remove(Config.lock_request_file)
                Config.lock_acquired = False
                Config.lock_requested = False
                print("Releasing the lock.")
        except OperationalError as e:
            loop_count += 1
            conn2 = get_db_connection(dbname=Config.ftune_target_db_name)
            cursor2 = conn2.cursor()
            continue
        # print('Exiting the loop')
        break
    

    cols = [desc[0] for desc in cursor.description]
    col_dtype_dict = get_column_dtype(table_name, cursor)
    all_str_vals = []
    str_cols = []
    print('des')

    total_rows = len(base_df)
    if removing_features:
        for feature in removing_features:
            if Config.do_string_encoding:
                if col_dtype_dict[feature] == 'character varying':
                    str_cols.append(feature)
                else:
                    df = df.drop(feature, axis=1)
                    base_df = base_df.drop(feature, axis=1)
            else:
                df = df.drop(feature, axis=1)
                base_df = base_df.drop(feature, axis=1)
    
    if len(str_cols) == 0:
        return base_df, df

    e = time.time()
    print(f'Loaded and checked table {table_name} in {e-s}s;\tstr_cols are: {str_cols}')
    
    for col in str_cols:
        df[col] = df[col].fillna('-far-')
        base_df[col] = base_df[col].fillna('-far-')

    
    global tokenizer, model, scaler, max_len, base_model_file_dir
    s = time.time()
    cf = Config.config_suffix
    ftune_idx = cf.index('_ftune')
    cf = cf[:ftune_idx]
    model = FastText.load(f"{base_model_file_dir}ftxt_{Config.db_name}_{table_name}_{col}_{cf}.model")
    e = time.time()
    print(f'\tLoaded the model in {e-s}s.')
        
    
    for col in str_cols:
        s = time.time()
        base_df[col] = base_df[col].apply(get_embedding)
        e = time.time()
        df2 = pd.DataFrame(base_df[col].to_list(), columns=[f'{col}_{i}' for i in range(embedding_dim)])
        base_df = pd.concat([base_df, df2], axis=1)
        base_df = base_df.drop(col, axis=1)
        e1 = time.time()
        print(f'\tEncoded {col} in {e-s}s, updated base df in {e1-e}')
    
    for col in str_cols:
        s = time.time()
        df[col] = df[col].apply(get_embedding)
        e = time.time()
        df2 = pd.DataFrame(df[col].to_list(), columns=[f'{col}_{i}' for i in range(embedding_dim)])
        df = pd.concat([df, df2], axis=1)
        df = df.drop(col, axis=1)
        e1 = time.time()
        print(f'\tEncoded {col} in {e-s}s, updated target df in {e1-e}')   

    return base_df, df


def get_embedding(new_string):
    global model
    return model.wv[new_string.lower()]

def get_embeddingmh(new_string):
    global embedding_dim
    m = MinHash(num_perm=embedding_dim)
    for token in new_string:
        m.update(token.encode('utf8'))
    return m.hashvalues


def get_embedding2(new_string):
    global tokenizer, model, scaler, max_len
    new_string = new_string.lower()
    new_sequence = tokenizer.texts_to_sequences([new_string])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len)
    new_padded_sequence = scaler.transform(new_padded_sequence)
    embedding = model.predict(new_padded_sequence, verbose=3)
    return embedding[0]


def pca_calculator(table_list, removing_features, precision, null_threshold=0.6, var_threshold=0.001, tables_pca = []):
    # success = []
    max_length = 0
    default_values = [-9999, -999]
    print('Calculating PC values for all tables:')
    if '_ftune' in Config.tb_encoding_method:
        Config.db_name = Config.ftune_base_db_name
        print(f'Changing db to {Config.db_name}')

    for table in tqdm(table_list):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            if 'sdss' in Config.db_name:
                if '_ftune' in Config.tb_encoding_method and 'NewMLPft' in Config.config_suffix:
                    df, target_df = get_table_data_ftune_fasttext(table, removing_features.get(table))
                else:
                    df = get_table_data(table, removing_features.get(table), cursor)
            elif '_ftune' in Config.tb_encoding_method and 'NewMLPft' in Config.config_suffix:
                print("opt1")
                df, target_df = smart_get_table_data_ftune_fasttext(table)
            else:
                df = smart_get_table_data(table, removing_features.get(table), cursor)

            if 'Normalize_NoPCA' in Config.tb_encoding_method:
                if table == 'tballbirds':
                    pass
                else:
                    df.fillna(0, inplace=True)
                    principal_df = pd.DataFrame(data=MinMaxScaler().fit_transform(df))
                    max_length = max(max_length, principal_df.shape[1])
                    tables_pca.append(TablePCA(table, principal_df))
                    continue

            if 'NoPCA' in Config.tb_encoding_method:
                df.fillna(0, inplace=True)
                tables_pca.append(TablePCA(table, df))
                continue

            # if table == 'platex':
            if 'sdss' in Config.db_name:
                for col in df.columns:
                    if df[col].isin(default_values).any():
                        min_val = df[col][(df[col] != -9999) & (df[col] != -999)].min()
                        df[col] = np.where((df[col] == -9999) | (df[col] == -999), min_val - 5, df[col])


            if 'PCAOnly' not in Config.tb_encoding_method:
                df, prep_summary = preprocess_data(table, df, null_threshold, var_threshold)
                if '_ftune' in Config.tb_encoding_method:
                    Config.db_name = Config.ftune_target_db_name
                    if 'NewMLPft' not in Config.config_suffix:
                        conn2 = get_db_connection()
                        cursor2 = conn2.cursor()
                        target_df = smart_get_table_data(table, removing_features.get(table), cursor2)
                    target_df, _ = preprocess_data(table, target_df, null_threshold, var_threshold, prev_summary=prep_summary, apply_prev_summary=True)
                    Config.db_name = Config.ftune_base_db_name


            if 'PCAOnly' in Config.tb_encoding_method:
                df.fillna(df.mean(), inplace=True) 
                precision = Config.encoding_length
                num_cols_to_add = Config.encoding_length - df.shape[1]
                if num_cols_to_add > 0:
                    row_mean = df.mean(axis=1)
                    print(f'Injecting {num_cols_to_add} columns to {table}')
                    additional_cols = pd.DataFrame({f'col_{i}': row_mean for i in range(num_cols_to_add+1)})
                    df = pd.concat([df, additional_cols], axis=1)
                df.fillna(0, inplace=True) 
                
            
            scalar = MinMaxScaler()
            # # scalar = StandardScaler()
            x = scalar.fit_transform(df)
                
            if  'evalPCAChange' not in Config.tb_encoding_method:
                if '_ftune' in Config.tb_encoding_method:
                    target_x = scalar.transform(target_df)
                if (table == 'tballstats7' or len(df.columns) > 36) and Config.db_name == 'genomic':
                    pca = PCA(n_components=36)
                else:
                    pca = PCA(n_components=precision)

                principal_components = pca.fit_transform(x)
                cPickle.dump(pca, open(f'SavedFiles/Models/PCA/{Config.db_name}_{table}_{Config.tb_encoding_method}_{Config.config_suffix}_PCA.pkl', 'wb'))
                if '_ftune' in Config.tb_encoding_method:
                    principal_components = pca.transform(target_x)

                principal_df = pd.DataFrame(data=MinMaxScaler().fit_transform(principal_components))
                tables_pca.append(TablePCA(table, principal_df, pca.explained_variance_, pca.components_))
                max_length = max(max_length, len(pca.explained_variance_ratio_))
                print(f'{table}: { len(pca.explained_variance_ratio_)}')
            else:
                if '_ftune' in Config.tb_encoding_method:
                    scalar2 = MinMaxScaler()
                    scalar2.fit(target_df) 
                    target_x = scalar2.transform(target_df)
                pca = IncrementalPCA(n_components=min(32, len(df.columns)))

                pca.partial_fit(x)
                pc1 = pca.components_.copy()
                cPickle.dump(pca, open(f'SavedFiles/Models/PCA/{Config.ftune_base_db_name}_{table}_{Config.tb_encoding_method}_{Config.config_suffix}_incPCA.pkl', 'wb'))
                print('dmuped PCA')
                if '_ftune' in Config.tb_encoding_method:
                    pca.partial_fit(target_x)
                    pc2 = pca.components_.copy()
                    cPickle.dump(pca, open(f'SavedFiles/Models/PCA/{Config.ftune_target_db_name}_Frm_{Config.ftune_base_db_name}_{table}_incPCA.pkl', 'wb'))

                    similarities = [np.dot(pc1[i], pc2[i]) /
                    (np.linalg.norm(pc1[i]) * np.linalg.norm(pc2[i]))
                    for i in range(len(pc1))]

                    with open(f'SavedFiles/Models/PCA/{Config.ftune_target_db_name}_Frm_{Config.ftune_base_db_name}_incPCA_summary.txt', 'a') as ofile:
                        ofile.write(f'{table}\nall similarities:\n\t')
                        ofile.write(", ".join(map(str, similarities)) + "\n")

                        ofile.write('min similarity:\n\t')
                        ofile.write(str(min(similarities)) + "\n")
                
        except Exception as e:
            print('exception for table {} with message: {}'.format(table, e))
            print(e.__repr__())
            traceback.print_exc()
            # break
            exit()
    
    print('max num of PCs is: ' + str(max_length))
    if '_ftune' in Config.tb_encoding_method:
        Config.db_name = Config.ftune_target_db_name
        print(f'Changing db back to {Config.db_name}')
    # exit()
    return tables_pca


def preprocess_data(tb, df, null_threshold, var_threshold, prev_summary=None, apply_prev_summary=False):
    prep_summary = {}
    if 'wiki' in Config.db_name:
        var_threshold /= 10

    if apply_prev_summary and prev_summary is not None:
        columns_to_drop = prev_summary['null_dropout']
        print(f'(Summary-based) {len(columns_to_drop)} NaN columns to drop from {len(df.columns)}')
        df = df.drop(columns=columns_to_drop)

        decimal_columns = prev_summary['decimal_cols']
        if 'benchbase' in Config.db_name or 'tpcc' in Config.db_name or 'tpch' in Config.db_name:
            for col in decimal_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        low_variance_cols = prev_summary['low_var_cols']
        print(f'(Summary-based) {len(low_variance_cols)} low variance columns to drop from {len(df.columns)}')
        df.drop(columns=low_variance_cols, inplace=True)

        df_mean = prev_summary['df_mean']
        df.fillna(df_mean, inplace=True) 

    else:
        null_percentages = df.isnull().mean()
        columns_to_drop = null_percentages[null_percentages > null_threshold].index.tolist()
        print(f'{len(columns_to_drop)} NaN columns to drop from {len(df.columns)}')
        df = df.drop(columns=columns_to_drop)

        decimal_columns = []
        for col in df.columns:
            if any(isinstance(val, Decimal) for val in df[col]):
                decimal_columns.append(col)
        if 'benchbase' in Config.db_name or 'tpcc' in Config.db_name or 'tpch' in Config.db_name:
            for col in decimal_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
        variances = df.var()  # or df.std()
        low_variance_cols = variances[variances < var_threshold].index
        print(f'{len(low_variance_cols)} low variance columns to drop from {len(df.columns)}')
        if len(df.columns) - len(low_variance_cols) <= 2:
            max_var = max(var for var in variances if var < var_threshold)
            low_variance_cols = variances[variances < max_var].index
            print(f'(Revised) - {len(low_variance_cols)} low variance columns to drop from {len(df.columns)}')

        df.drop(columns=low_variance_cols, inplace=True)

        df_mean = df.mean()
        df.fillna(df_mean, inplace=True) 

        prep_summary['null_dropout'] = columns_to_drop
        prep_summary['decimal_cols'] = decimal_columns
        prep_summary['low_var_cols'] = low_variance_cols
        prep_summary['df_mean'] = df.mean()

    return df, prep_summary

def get_pca_table_info():
    if Config.db_name == 'tpcds':
        tables = [item for item in Config.table_list]
        feature_exclude_dict = {}
        json_file_path = Config.base_dir + 'Data/tb_col_summary.json'
        with open(json_file_path, 'r') as json_file:
            feature_exclude_dict = json.load(json_file)
    elif Config.db_name in ['genomic', 'birds']:
        tables = [item for item in Config.table_list]
        feature_exclude_dict = {}
    else:
        tables = [item for item in Config.table_list if item not in Config.pca_exclude_tables]
        feature_exclude_dict = {}
    return tables, feature_exclude_dict


def main():
    tables, feature_exclude_dict = get_pca_table_info()
    pca_calculator(tables, feature_exclude_dict, 0.9)


if __name__ == '__main__':
    main()
    print('done')
