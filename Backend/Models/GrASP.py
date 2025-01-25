import os
import sys
import tensorflow as tf
from tensorflow import keras

from typing import List

from keras.layers import Input, LSTM, Dense, RepeatVector, Reshape, Flatten, \
     Dropout, Conv1D, GlobalAveragePooling1D, Softmax
from keras.models import Model, model_from_json
from keras.utils import Sequence


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from Backend.Database import helpers as db_helper
from Backend.Database.LRUCache import LRUCache
from Backend.Util import PartitionManager, AffinityMatrix
from Backend.Util.BackendUtilFunctions import get_encoded_block_aggregation, get_sf, get_par_tb_deltas
from Backend.Util.LogComp import LogLine
from Backend.Util.PartitionManager import Partition
from Backend.Util.TableManager import TableManager
from Configuration.config import Config, alter_config
from Utils.utilFunction import get_log_lines, get_complete_query_result_details3
from main import create_table_manager, create_par_manager_and_aff_matrix, get_tables_bid_range, \
    get_tables_actual_bid_range, create_pm_af_for_configs, get_dataset_workloads, my_to_csv, get_db_pretest_queries
import pandas as pd
import numpy as np
import math
import csv
csv.field_size_limit(100 * 1024 * 1024) 
import _pickle as cPickle
import time
from tqdm import tqdm
import pprint as pp
import re
import traceback
from collections import Counter
import matplotlib.pyplot as plt
import random
plt.rcParams['font.family'] = 'Times New Roman'
import statistics

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


def get_managers(config_suffix, read_tb_manager=1, read_par_manager=1, read_aff_matrix=1):
    if config_suffix == '':
        table_manager: TableManager = create_table_manager(
            read_tb_manager, Config.encoding_length, Config.encoding_epoch_no,
        )  
        print(table_manager.tables.keys())

        p_, a_ = create_par_manager_and_aff_matrix(
            read_par_manager=read_par_manager, read_aff_matrix=read_aff_matrix,
            table_manager=table_manager
        )
        partition_manager: PartitionManager.PartitionManager = p_
        aff_matrix: AffinityMatrix.AffinityMatrix = a_

    else:
        print('Getting the components')        
        table_manager: TableManager = create_table_manager(read_tb_manager, Config.encoding_length, Config.encoding_epoch_no,
                        f'{Config.base_dir}{Config.db_name}table_manager{config_suffix}B{Config.logical_block_size}P{Config.max_partition_size}.p')
        p_, a_ = create_pm_af_for_configs(
            read_par_manager, read_aff_matrix,
            table_manager=table_manager, suffix=config_suffix
        )
        partition_manager: PartitionManager.PartitionManager = p_
        aff_matrix: AffinityMatrix.AffinityMatrix = a_

    return table_manager, partition_manager, aff_matrix

def store_model(model, model_name):
    model_json = model.to_json()
    with open(f"{base_model_file_dir}{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f"{base_model_file_dir}{model_name}.h5")
    print(f"Saved {model_name} to disk")

def load_model(model_name):
    json_file = open(f'{base_model_file_dir}{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f"{base_model_file_dir}{model_name}.h5")
    print(f"Loaded {model_name} from disk")
    return loaded_model

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size    
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X[0]) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = []
        y_batch = []
        for i in range(len(self.X)):
            X_batch.append(self.X[i][indexes])
            
        for i in range(len(self.y)):
            y_batch.append(self.y[i][indexes])
        
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X[0]))
        if self.shuffle:
            np.random.shuffle(self.indexes)

class GrASP_comb():
    def __init__(self, model_name, res_file_name, config_suffix, table_manager, partition_manager, aff_mat, output_type, 
                 pid_selection_mode='last', size_type='class', pred_count_limit =50, cache_size=66000, query_template_encoding_method='sql', received_args={}):
        self.model_name = model_name
        self.res_file_name = res_file_name
        self.config_suffix = config_suffix
        self.table_manager = table_manager
        self.partition_manager: PartitionManager.PartitionManager = partition_manager
        self.aff_mat = aff_mat
        self.output_type = output_type
        self.pid_selection_mode = pid_selection_mode
        self.cache_size = cache_size
        self.look_back = Config.look_back
        self.rows = len(Config.table_list)
        self.cols = Config.encoding_length
        self.file_name = f'{Config.db_name}_all_train{config_suffix}WB{Config.logical_block_size}WP{Config.max_partition_size}'
        if 'tpch_sf10' in Config.db_name and Config.max_partition_size == 32:
            self.file_name = f'{Config.db_name}_all_trainf{config_suffix}WB{Config.logical_block_size}WP{Config.max_partition_size}'

        self.delta_classes = {}
        self.delta_classes_reverse = {}
        self.query_template_encoding_method = query_template_encoding_method

        if Config.db_name == 'birds':
            self.pred_count_limit = 60
            self.delta_class_count_limit = 500
        else:
            self.pred_count_limit = 120
            self.delta_class_count_limit = 1500
        
        self.delta_class_mod = 1
        mod_pattern = r'_mod(\d*)_'
        mod_matches = re.findall(mod_pattern, self.model_name)
        print(mod_matches)
        if len(mod_matches):
            print(f'Generating delta classes with mod {mod_matches[-1]}')
            self.delta_class_mod = int(mod_matches[-1])

        mod_pattern = r'_dc(\d*)_'
        mod_matches = re.findall(mod_pattern, self.model_name)
        print(mod_matches)
        if len(mod_matches):
            print(f'Change delta class limit to {mod_matches[-1]}')
            self.delta_class_count_limit = int(mod_matches[-1])

        self.qt_encod_size = 32
        if 'qjsn' in self.model_name:
            if 'qjsn_simple' in self.model_name:
                self.qt_encod_size = 4 + len(Config.table_list)
            else:    
                self.qt_encod_size = 4 + len(Config.table_list)*17
        self.freq_tb_based_deltas = {}
  
    def get_delta_classes(self, all_deltas_dict):
        combined_counter = Counter()
        for tb_idx in range(len(Config.table_list)):
            tb = Config.table_list[tb_idx]
            all_deltas = all_deltas_dict[tb]
            if self.delta_class_mod > 1:
                all_deltas = [int(d/self.delta_class_mod) for d in all_deltas]
            
            counter = Counter(all_deltas)
            if len(counter.items()) == 0:
                print(f'no delta for {tb}')

            deltas_with_gt1_frequency = [delta for delta, count in counter.items() if count > 5]

            self.freq_tb_based_deltas[tb] = deltas_with_gt1_frequency
            combined_counter = combined_counter + counter

        all_delta_cnt = len(combined_counter)
        tb_delta_class_count = int(1.5 * (min(self.delta_class_count_limit, all_delta_cnt))) + 1

        print(f'selecting {tb_delta_class_count} deltas from all {all_delta_cnt} deltas')
        top_deltas = combined_counter.most_common(tb_delta_class_count - 1) #-1 is because of the zero

        pref_log_file = f'{pref_log_file_base}/{Config.db_name}_total_deltasWB{Config.logical_block_size}WP{Config.max_partition_size}.csv'
        with open(pref_log_file, 'a') as ofile:
            ofile.write(f"{self.model_name}: {all_delta_cnt}")

        for i in range(1, len(top_deltas)+1): # zero is reserved for unfrequent ones
            self.delta_classes[top_deltas[i-1][0]] = i
            self.delta_classes_reverse[i] = top_deltas[i-1][0]
        
        self.delta_class_count = len(top_deltas) + 1
        mod_pattern = r'_dc(\d*)_'
        mod_matches = re.findall(mod_pattern, self.model_name)
        print(mod_matches)
        if len(mod_matches) == 0:
            added_delta = 1250
        else:
            added_delta = int(mod_matches[-1])
        if '_loadDelC' in self.model_name:
            self.delta_class_count = added_delta
        elif '_addDelC' in self.model_name:
            self.delta_class_count += added_delta
        print(f'delta_class_count = {self.delta_class_count}')

    def get_query_tb_based_res(self, par_dict):
        res = []
        sorted_tb = sorted(par_dict.keys(), key=lambda x: Config.table_list.index(x))
        for tb in sorted_tb:
            for delta in par_dict[tb]:
                res.append(f'{tb}_{delta}')
        return res
    
    def get_binary_delta_class(self, delta_seq, stat_on=False):
        global NON_EXISTING_DELTAS
        result_array = np.zeros(self.delta_class_count, dtype=int)
        sum_len = 0
        for tb_idx in range(len(Config.table_list)):
            tb = Config.table_list[tb_idx]
            if tb in delta_seq:
                indexes = []
                for delta in delta_seq[tb]:
                    if delta in self.delta_classes:
                        if 'DelC' in self.model_name and self.delta_classes[delta] >= self.delta_class_count:
                            continue
                        indexes.append(self.delta_classes[delta])
                    elif stat_on:
                        NON_EXISTING_DELTAS += 1

                result_array[indexes] = 1
                sum_len += len(indexes)
        
        if sum_len == 0:
            # There was no existing delta in this query. Set the default delta class
            result_array[[0]] = 1
        return result_array   

    def get_par_deltas_per_table(self, prev_par_dict, par_dict, no_prev=False):
        if '_par_based_delta' in self.output_type:
            delta_matrix = {}
            for tb_idx in range(len(Config.table_list)):
                tb = Config.table_list[tb_idx]
                if tb in par_dict:
                    if prev_par_dict is None or no_prev:
                        last_accessed_pid = 0
                    else:
                        try:
                            last_accessed_pid = self.get_base_pnumber(prev_par_dict)
                        except Exception as e:
                            err = traceback.format_exc()
                            print(err)
                            print(prev_par_dict)
                            input('enter')
                            continue
                    
                    deltas = [second - last_accessed_pid for second in par_dict[tb]]
                    delta_matrix[tb] = deltas

        return delta_matrix
    
    def get_base_pnumber(self, prev_par_dict):
        sorted_tb = sorted(prev_par_dict.keys(), key=lambda x: Config.table_list.index(x))
        if 'first' in self.output_type:
            return prev_par_dict[sorted_tb[0]][0]
        elif 'last' in self.output_type:
            return prev_par_dict[sorted_tb[-1]][-1]
        elif 'mid' in self.output_type:
            all_deltas = []
            for key in prev_par_dict:
                all_deltas.extend([f'{key}_{delt}' for delt in prev_par_dict[key]])
            sorted_deltas = sorted(all_deltas, key=lambda x: (Config.table_list.index(x.rsplit('_', 1)[0]), int(x.rsplit('_', 1)[1])))
            return int(sorted_deltas[len(sorted_deltas) // 2].rsplit('_', 1)[1])
        else:
            print('no match for output type, returning zero')
            return [0]
    
    def get_last_acc_tb(self, tb_list):
        sorted_tb = sorted(tb_list, key=lambda x: Config.table_list.index(x))
        return sorted_tb[-1]
    
    def prepare_train_data(self, just_get_deltas=False, up_to='the end'):
        prep_time = 0
        print('Preparing the data')
        sep = Config.csv_file_separator
        print(f'reading train data from {self.file_name}.txt with sep = {sep}')
        train_df = pd.read_csv(
            f'./Data/{self.file_name}.txt',
            header=0,
            sep=sep,
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            engine='python'
        )

        if QLIMIT != -1:
            train_df = train_df[:min(len(train_df), QLIMIT)].copy()
                

        if 'ClientIP' in train_df.columns:
            train_df.rename(columns={'ClientIP': 'clientIP'}, inplace=True)

        if Config.db_name in ['genomic', 'birds']:  
            train_df['qPlan'] = '-' 
        if (Config.db_name not in ['genomic', 'birds']) and( 'qPlan' not in train_df.columns or '-' in train_df['qPlan'].to_list()):  
            print(f'Getting the query plans. size of df before {len(train_df)}')
            train_df['qPlan'] = '-'
            conn = db_helper.get_db_connection()
            cursor = conn.cursor()
            
            for idx, row in train_df.iterrows():
                stmt = row['statement'].lower()
                explain_query = f'EXPLAIN (COSTS off) {stmt}'
                cursor.execute(explain_query)
                plan_rows = cursor.fetchall()
                plan_text = ' '.join(row[0] for row in plan_rows if 'Workers Planned' not in row[0])
                train_df.loc[idx, 'qPlan'] = plan_text.replace(sep, ' ')
            print(f'size of new df after {len(train_df)}')
            my_to_csv(train_df, f'./Data/{self.file_name}.txt', sep=sep, remove_quotes=False)
        
        if ('qJsonPlan'not in train_df.columns or '-' in train_df['qJsonPlan'].to_list() or IS_ANALYZING_TIME) and 'qjsn' in self.model_name:  
            print(f'Getting the query json plans. size of df before {len(train_df)}')
            train_df['qJsonPlan'] = '-'
            conn = db_helper.get_db_connection()
            cursor = conn.cursor()

            t1 = time.time()
            
            for idx, row in train_df.iterrows():
                stmt = row['statement'].lower()
                explain_query = f'EXPLAIN (COSTS off, FORMAT JSON) {stmt}'
                cursor.execute(explain_query)
                plan_rows = cursor.fetchall()
                train_df.loc[idx, 'qJsonPlan'] = json.dumps(plan_rows[0][0])

            t2 = time.time()
            prep_time += t2 - t1
            # print(train_df.head())
            print(f'size of new df after {len(train_df)}')
            if not IS_ANALYZING_TIME:
                my_to_csv(train_df, f'./Data/{self.file_name}.txt', sep=sep, remove_quotes=False)


        train_df['encResultBlock'] = '-'
        train_df['partitionNumbers_perTb'] = '-'
        train_df['partitionNumbers'] = train_df['resultPartitions'].apply(lambda x: sorted([int(re.search(r'\d+', item).group()) for item in x.split(',')]))
        self.max_p_count = train_df['partitionNumbers'].apply(len).max() + 1
        prepration_time_start = time.time() - prep_time
        t1 = time.time()
        train_df = get_complete_query_result_details3(train_df, self.partition_manager, pid_selection_mode=self.pid_selection_mode)
        self.max_p_count_of_tbs = {}
        for tb in Config.table_list:
            self.max_p_count_of_tbs[tb] = int(1.2 * train_df['partitionNumbers_perTb'].apply(lambda d: len(d[tb]) if tb in d else 0).max()) + 1

        if 'wqtenc' in self.model_name:
            if 'qjsn' in self.model_name:
                qtenc_model_file_name = f'./SavedFiles/Models/qt_encoders/{self.file_name}_{self.query_template_encoding_method}_qtj_preprocL{PLAN_PREP_LEVEL}.model'
                try:
                    train_df['processedJPlan'] = '-'
                    all_join_conditions, all_filter_conditions = [], []
                    for idx, row in train_df.iterrows():
                        stmt = row['statement'].lower()
                        jplan = json.loads(row['qJsonPlan'])
                        q_type_repr, tables, join_conditions, filter_conditions = self.encode_json_plan(stmt, jplan)

                        alias_refs = {}                    
                        for tb, alias in tables:
                            alias_refs[alias] = tb

                        join_conditions_per_tb = {}
                        for ac_idx, cond in enumerate(join_conditions):
                            for alias in alias_refs:
                                if alias in cond:
                                    tb = alias_refs[alias]
                                    curr_cond = join_conditions_per_tb.get(tb, '')
                                    join_conditions_per_tb[tb] = f'{curr_cond} {cond}'
                        
                        if len(join_conditions_per_tb):
                            all_join_conditions.extend(list(join_conditions_per_tb.values()))

                        filter_conditions_per_tb = {}
                        for ac_idx, (alias, cond) in enumerate(filter_conditions):
                            tb = alias_refs[alias]
                            curr_cond = filter_conditions_per_tb.get(tb, '')
                            filter_conditions_per_tb[tb] = f'{curr_cond} {cond}'
                        
                        if len(filter_conditions_per_tb):
                            all_filter_conditions.extend(list(filter_conditions_per_tb.values()))

                        train_df.loc[idx, 'processedJPlan'] = json.dumps([q_type_repr, list(alias_refs.values()), join_conditions_per_tb, filter_conditions_per_tb, stmt])
                
                    if 'qjsn_simple' not in self.model_name:
                        qt_encoder = Doc2Vec.load(qtenc_model_file_name)
                        self.qt_encoder = qt_encoder
                    else:
                        qt_encoder = Doc2Vec(vector_size=8, min_count=1, epochs=25)

                except FileNotFoundError:
                    if 'qjsn_nlp' in self.model_name:
                        jall_stmts = [preprocess_plan_text(stmt) for stmt in set(train_df['statement'].to_list())]
                    else:
                        jall_stmts = [preprocess_plan_text(stmt) for stmt in set(all_join_conditions + all_filter_conditions)]
                    # print(jall_stmts[:5])
                    jdocuments = [
                        TaggedDocument(words=plan_tokens, tags=[f'cond_{i}'])
                        for i, plan_tokens in enumerate(jall_stmts)
                    ]
                    qt_encoder = Doc2Vec(vector_size=8, min_count=1, epochs=25)
                    qt_encoder.build_vocab(jdocuments)
                    qt_encoder.train(jdocuments, total_examples=qt_encoder.corpus_count, epochs=qt_encoder.epochs)
                    qt_encoder.save(qtenc_model_file_name)
                    self.qt_encoder = qt_encoder

                if up_to == 'qt_enc':
                    return
                train_df['qt_enc'] = train_df['processedJPlan'].apply(self.get_query_representation)

            else:                
                qtenc_model_file_name = f'./SavedFiles/Models/qt_encoders/{self.file_name}_{self.query_template_encoding_method}_preprocL{PLAN_PREP_LEVEL}.model'
                try:
                    qt_encoder = Doc2Vec.load(qtenc_model_file_name)
                    if self.query_template_encoding_method == 'sql':
                        train_df['prep_stmt'] = train_df['statement'].apply(process_plan_text)
                    else:
                        train_df['prep_stmt'] = train_df['qPlan'].apply(process_plan_text)
                    train_df['prep_stmt'] = train_df['prep_stmt'].apply(preprocess_plan_text)
                    self.qt_encoder = qt_encoder
                except FileNotFoundError as fe:
                    train_df['prep_stmt'] = train_df['statement'].apply(process_plan_text)
                    all_stmts = [preprocess_plan_text(stmt) for stmt in set(train_df['prep_stmt'].to_list())]
                    train_df['prep_stmt'] = train_df['prep_stmt'].apply(preprocess_plan_text)

                    documents = [
                        TaggedDocument(words=plan_tokens, tags=[f'plan_{i}'])
                        for i, plan_tokens in enumerate(all_stmts)
                    ]
                    qt_encoder = Doc2Vec(vector_size=32, min_count=1, epochs=40)
                    qt_encoder.build_vocab(documents)
                    qt_encoder.train(documents, total_examples=qt_encoder.corpus_count, epochs=qt_encoder.epochs)
                    qt_encoder.save(qtenc_model_file_name)
                    self.qt_encoder = qt_encoder
                
                if up_to == 'qt_enc':
                    return
                train_df['qt_enc'] = train_df['prep_stmt'].apply(self.get_query_template_enc)

        # Calculate train_df sequence of encoded result set and the binary result set
        encoded_seqs = []
        binary_output_seqs = []
        res_tbs_seqs = []
        res_pcount_seqs = []
        qt_enc_seqs = []
        last_tb_seqs = []
        deltas_seqs = []
        all_deltas = {}
        for tb in Config.table_list:
            all_deltas[tb] = []

        q_delta = self.get_par_deltas_per_table(None, train_df.loc[0, 'partitionNumbers_perTb'])
        res_tbs_seq = [get_encoded_res_tbs(train_df.loc[0, 'resultTables'])]
        res_pcount_seq = [[1 if ii == min(train_df.loc[0, 'resultParCount'], self.pred_count_limit) else 0 for ii in range(1, self.pred_count_limit+1)]]
        deltas_seq = [q_delta]
        if 'wqtenc' in self.model_name:
            qt_enc_seq = [train_df.loc[0, 'qt_enc']]
            last_tb_seq = [[0 if tbt != self.get_last_acc_tb(train_df.loc[0, 'resultTables']) else 1 for tbt in Config.table_list]]
        for tb in q_delta:
            all_deltas[tb].extend(q_delta[tb])
        # all_deltas = q_delta
        enc_seq = [get_encoded_block_aggregation(train_df.loc[0, 'encResultBlock'], False)]
        for i in range(1, len(train_df)):
            if train_df.loc[i, 'encResultBlock'] == '-':
                continue
            if train_df.loc[i, 'clientIP'] == train_df.loc[i - 1, 'clientIP']:
                enc_seq.append(get_encoded_block_aggregation(train_df.loc[i, 'encResultBlock'], False))
                q_delta = self.get_par_deltas_per_table(train_df.loc[i-1, 'partitionNumbers_perTb'], train_df.loc[i, 'partitionNumbers_perTb'])
                deltas_seq.append(q_delta)
                res_tbs_seq.append(get_encoded_res_tbs(train_df.loc[i, 'resultTables']))
                res_pcount_seq.append([1 if ii == min(train_df.loc[i, 'resultParCount'], self.pred_count_limit) else 0 for ii in range(1, self.pred_count_limit+1)])
                if 'wqtenc' in self.model_name:
                    qt_enc_seq.append(train_df.loc[i, 'qt_enc'])
                    last_tb_seq.append([0 if tbt != self.get_last_acc_tb(train_df.loc[i, 'resultTables']) else 1 for tbt in Config.table_list])
                # all_deltas = np.concatenate(all_deltas, q_delta)
                for tb in q_delta:
                    all_deltas[tb].extend(q_delta[tb])
            elif len(enc_seq) > 0:
                encoded_seqs.append(enc_seq)
                deltas_seqs.append(deltas_seq)
                res_pcount_seqs.append(res_pcount_seq)
                res_tbs_seqs.append(res_tbs_seq)
                if 'wqtenc' in self.model_name:
                    qt_enc_seqs.append(qt_enc_seq)
                    last_tb_seqs.append(last_tb_seq)
                enc_seq = [get_encoded_block_aggregation(train_df.loc[i, 'encResultBlock'], False)]
                q_delta = self.get_par_deltas_per_table(None, train_df.loc[i, 'partitionNumbers_perTb'])
                deltas_seq = [q_delta]
                res_tbs_seq = [get_encoded_res_tbs(train_df.loc[i, 'resultTables'])]
                res_pcount_seq = [[1 if ii == min(train_df.loc[i, 'resultParCount'], self.pred_count_limit) else 0 for ii in range(1, self.pred_count_limit+1)]]
                if 'wqtenc' in self.model_name:
                    qt_enc_seq = [train_df.loc[i, 'qt_enc']]
                    last_tb_seq = [[0 if tbt != self.get_last_acc_tb(train_df.loc[i, 'resultTables']) else 1 for tbt in Config.table_list]]
                # all_deltas = np.concatenate(all_deltas, q_delta)
                for tb in q_delta:
                    all_deltas[tb].extend(q_delta[tb])
        if len(enc_seq) > 0:
            encoded_seqs.append(enc_seq)
            deltas_seqs.append(deltas_seq)
            res_pcount_seqs.append(res_pcount_seq)
            res_tbs_seqs.append(res_tbs_seq)
            if 'wqtenc' in self.model_name:
                qt_enc_seqs.append(qt_enc_seq)
                last_tb_seqs.append(last_tb_seq)


        t2 = time.time()
        prep_time += t2 - t1
        set_all_deltas = np.unique(np.concatenate(list(all_deltas.values())))
        if just_get_deltas:
            return all_deltas
        if '_par_based_delta' in self.output_type:
            t1 = time.time()
            self.get_delta_classes(all_deltas)
            t2 = time.time()
            delta_selection_time = t2 - t1
            prep_time += t2 - t1

        t1 = time.time()
        if 'minmax' in self.model_name:
            self.deltas_mean = min(set_all_deltas)
            self.deltas_std = max(set_all_deltas) - self.deltas_mean
        else:
            self.deltas_mean = np.mean(set_all_deltas)
            self.deltas_std = np.std(set_all_deltas)


        if 'biDelta' in self.model_name:
            self.max_p_count = self.delta_class_count

        # Convert the sequences to actual input/output for the model
        data_x, data_y = [], []
        data_y_res_tbs = []
        data_y_res_pcount = []
        data_x_delta = []
        data_x_res_pcount = []
        data_x_res_tbs = []
        data_x_qt_enc = []
        data_x_last_tb = []
        for j in range(len(encoded_seqs)):
            # for j in range(5):
            for i in range(len(encoded_seqs[j]) - self.look_back):
                data_x.append(np.array(encoded_seqs[j][i:(i + self.look_back)]))
                if 'all_hists' in self.model_name:
                    hist_deltas = []
                    for ih in range(self.look_back):
                        r_deltas = []
                        step_delta = deltas_seqs[j][i+ih]
                        if 'biDelta' in self.model_name:
                            normalized_delta = self.get_binary_delta_class(step_delta)
                        else:
                            for tb in Config.table_list:
                                if tb in step_delta:
                                    normalized_delta = (np.array(step_delta[tb]) - self.deltas_mean) / self.deltas_std
                                    if self.max_p_count_of_tbs[tb] - len(normalized_delta) > 0:
                                        normalized_delta = np.concatenate([normalized_delta, np.zeros(self.max_p_count_of_tbs[tb] - len(normalized_delta))])
                                else:
                                    normalized_delta = np.zeros(self.max_p_count_of_tbs[tb])       
                                r_deltas.append(normalized_delta)

                            normalized_delta = np.concatenate(r_deltas)
                        hist_deltas.append(normalized_delta)
                        
                    data_x_delta.append(np.array(hist_deltas))
                    data_x_res_tbs.append(np.array(res_tbs_seqs[j][i:(i + self.look_back)]))
                    data_x_res_pcount.append(np.array(res_pcount_seqs[j][i:(i + self.look_back)]))
                    if 'wqtenc' in self.model_name:
                        data_x_qt_enc.append(np.array(qt_enc_seqs[j][i:(i + self.look_back)]))
                        data_x_last_tb.append(np.array(last_tb_seqs[j][i:(i + self.look_back)]))
                else:
                    r_deltas = []
                    step_delta = deltas_seqs[j][i+self.look_back-1]
                    if 'biDelta' in self.model_name:
                        normalized_delta = self.get_binary_delta_class(step_delta)
                    else:
                        for tb in Config.table_list:
                            if tb in step_delta:
                                normalized_delta = (np.array(step_delta[tb]) - self.deltas_mean) / self.deltas_std
                                if self.max_p_count_of_tbs[tb] - len(normalized_delta) > 0:
                                    normalized_delta = np.concatenate([normalized_delta, np.zeros(self.max_p_count_of_tbs[tb] - len(normalized_delta))])
                            else:
                                normalized_delta = np.zeros(self.max_p_count_of_tbs[tb])       
                            r_deltas.append(normalized_delta)

                    normalized_delta = np.concatenate(r_deltas)
                    
                    data_x_delta.append(normalized_delta)
                    data_x_res_tbs.append(res_tbs_seqs[j][i+self.look_back-1])
                    data_x_res_pcount.append(res_pcount_seqs[j][i+self.look_back-1])
                    if 'wqtenc' in self.model_name:
                        if '_mixenc' in self.model_name:
                            data_x_qt_enc.append(np.array(qt_enc_seqs[j][i:(i+self.look_back)]))
                        else:
                            data_x_qt_enc.append(np.array(qt_enc_seqs[j][i+self.look_back-1]))
                        data_x_last_tb.append(np.array(last_tb_seqs[j][i+self.look_back-1]))

                if self.output_type == 'consec_delta_simple':
                    normalized_delta_y = np.array(deltas_seqs[j][i+self.look_back])
                    if len(normalized_delta_y) > self.pred_count_limit:
                        normalized_delta_y = normalized_delta_y[:self.pred_count_limit]
                    elif len(normalized_delta_y) < self.pred_count_limit:
                        normalized_delta_y = np.concatenate([normalized_delta_y, np.zeros(self.pred_count_limit - len(normalized_delta_y))])
                elif '_par_based_delta' in self.output_type:
                    normalized_delta_y = []
                    # for delta in deltas_seqs[j][i+self.look_back]:
                    normalized_delta_y = self.get_binary_delta_class(deltas_seqs[j][i+self.look_back])
                    normalized_delta_y = np.array(normalized_delta_y)
                data_y.append(normalized_delta_y)
                data_y_res_pcount.append(res_pcount_seqs[j][i+self.look_back])
                data_y_res_tbs.append(res_tbs_seqs[j][i+self.look_back])
                    

        # Make the model input and output
        t2 = time.time()
        prep_time += t2 - t1
        data_x = np.array(data_x)
        self.data_x_delta = np.expand_dims(np.array(data_x_delta), axis=1)
        self.data_y = np.array(data_y)
        print(data_x.shape)
        self.data_x = data_x.reshape(data_x.shape[0], self.look_back, data_x.shape[2] * data_x.shape[3])
        self.data_x_res_tbs = np.expand_dims(np.array(data_x_res_tbs), axis=1)
        self.data_x_qt_enc = np.expand_dims(np.array(data_x_qt_enc), axis=1)
        self.data_x_last_tb = np.expand_dims(np.array(data_x_last_tb), axis=1)
        self.data_x_res_pcount = np.expand_dims(np.array(data_x_res_pcount), axis=1)
        self.data_y_res_pcount = np.array(data_y_res_pcount)
        self.data_y_res_tbs = np.array(data_y_res_tbs)
        print(self.data_x.shape)
        print(self.data_x_res_tbs.shape)
        print(self.data_x_last_tb.shape)
        print(self.data_x_qt_enc.shape)
        print(self.data_x_res_pcount.shape)
        print(self.data_x_delta.shape)
        print(self.data_y.shape)


        all_info_dict = {
            "delta_classes": self.delta_classes,
            "delta_classes_reverse": self.delta_classes_reverse,
            "max_p_count": self.max_p_count,
            "max_p_count_of_tbs": self.max_p_count_of_tbs,
            "delta_class_count": self.delta_class_count,
            "deltas_mean": self.deltas_mean,
            "deltas_std": self.deltas_std,
            "pred_count_limit": self.pred_count_limit
        }

        if 'grasp' in self.model_name:
            all_info_dict["freq_tb_based_deltas"] = self.freq_tb_based_deltas
        
        cPickle.dump(all_info_dict, open(f"{base_model_file_dir}{self.model_name}{FQTESTSUFFIX}_all_info_dict.p", 'wb'))

        prepration_time_end = time.time()
        if IS_ANALYZING_TIME:
            with open(f'{result_base_path}/logFiles/{Config.db_name}_log_data.txt', 'a') as outfi:
                outfi.write(f'\nmodel_{self.model_name} total prep time: {prepration_time_end-prepration_time_start}')
                outfi.write(f'\nmodel_{self.model_name} prep time: {prep_time}')
                outfi.write(f'\nmodel_{self.model_name} delta selection time: {delta_selection_time}')

    def create_model(self):
        delta_dim = sum(list(self.max_p_count_of_tbs.values()))
        if use_binary_delta:
            delta_dim = self.delta_class_count
        n_enc_unit = 64
        n_unit_l = 32
        n_unit = 64
        rows = len(Config.table_list)
        # cols = 2 * Config.encoding_length
        cols = Config.encoding_length
        data_shape = rows * cols
        encoder_input = Input(shape=(rows, cols), name='qenc_encoder_in')
        enc_h1 = Flatten()(encoder_input)
        enc_out = Dense(units=n_enc_unit, activation='relu', name='qenc_dense')(enc_h1)
        conv_encoder = Model(inputs=encoder_input, outputs=enc_out)

        delta_encoder_input = Input(shape=(delta_dim,), name='delta_enc_input')
        enc_h12 = Flatten()(delta_encoder_input)
        delta_enc_out = Dense(units=n_unit, activation='relu', name='delta_dense')(enc_h12)
        delta_enc_out = tf.keras.layers.Flatten()(delta_enc_out)   
        delta_encoding = Reshape(target_shape=(1, n_unit))(delta_enc_out) 
        delta_encoder = Model(inputs=delta_encoder_input, outputs=delta_encoding)

        pcount_enc_input = Input(shape=(self.pred_count_limit,), name='pcountenc_in')
        pcount_enc_out = Dense(units=n_unit_l, activation='relu', name='pcount_dense')(pcount_enc_input)
        pcount_enc_out = Flatten()(pcount_enc_out)
        pcount_encoding = Reshape(target_shape=(1, n_unit_l), name='pcount_enc_reshape')(pcount_enc_out) 
        pcount_encoder = Model(inputs=pcount_enc_input, outputs=pcount_encoding)

        tbs_enc_input = Input(shape=(len(Config.table_list),), name='tbsenc_in')
        res_tbs_enc_outt = Dense(units=n_unit_l, activation='relu', name='restbs_dense')(tbs_enc_input)
        res_tbs_enc_out = Flatten()(res_tbs_enc_outt)
        res_tbs_encoding = Reshape(target_shape=(1, n_unit_l), name='tb_enc_reshape')(res_tbs_enc_out) 
        tbs_encoder = Model(inputs=tbs_enc_input, outputs=res_tbs_encoding)

        if 'wqtenc' in self.model_name:
            last_acc_tb_in = Input(shape=(len(Config.table_list),), name='lasttbsenc_in')
            last_acc_tb_outt = Dense(units=n_unit_l, activation='relu', name='lasttb_dense')(last_acc_tb_in)
            last_acc_tb_out = Flatten()(last_acc_tb_outt)   
            last_acc_tboding = Reshape(target_shape=(1, n_unit_l), name='lasttb_enc_reshape')(last_acc_tb_out) 
            last_acc_tb_encoder = Model(inputs=last_acc_tb_in, outputs=last_acc_tboding)

            qt_encoder_input = Input(shape=(self.qt_encod_size,), name='qt_enc_input')
            qenc_h12 = Flatten()(qt_encoder_input)
            qt_enc_out = Dense(units=n_unit_l, activation='relu', name='qt_dense')(qenc_h12)
            qt_enc_out = tf.keras.layers.Flatten()(qt_enc_out)   
            qt_encoding = Reshape(target_shape=(1, n_unit_l))(qt_enc_out) 
            qt_encoder = Model(inputs=qt_encoder_input, outputs=qt_encoding)



        # Model input
        model_input = Input(shape=(self.look_back, data_shape), name='qenc_input')
        reshape_input = Reshape((self.look_back, rows, cols))(model_input)
        
        if 'all_hists' in self.model_name:
            #delta_input
            delta_input = Input(shape=(1,self.look_back,delta_dim), name='deltaenc_in')

            #pcount input
            res_pcount_input = Input(shape=(1,self.look_back,self.pred_count_limit), name='pcountenc_in')

            #res tbs input
            res_tbs_input = Input(shape=(1, self.look_back, len(Config.table_list)), name='tbsenc_in')

            if 'wqtenc' in self.model_name:
                qtenc_input = Input(shape=(1, self.look_back, self.qt_encod_size), name='qtenc_in')
                last_acc_tb_input = Input(shape=(1, self.look_back, len(Config.table_list)), name='lasttb_in')


        else:
            #delta_input
            delta_input = Input(shape=(1,delta_dim), name='deltaenc_in')

            #pcount input
            res_pcount_input = Input(shape=(1,self.pred_count_limit), name='pcountenc_in')

            #res tbs input
            res_tbs_input = Input(shape=(1, len(Config.table_list)), name='tbsenc_in')

            if 'wqtenc' in self.model_name:
                qtenc_input = Input(shape=(1, self.qt_encod_size), name='qtenc_in')
                last_acc_tb_input = Input(shape=(1, len(Config.table_list)), name='lasttb_in')
            
        encoded_matrices = []

        for i in range(self.look_back):
            print(i)
            step_encoddings = []
            enc_out = conv_encoder(reshape_input[:, i, :, :])  # Encode each matrix separately
            step_encoddings.append(enc_out)

            if 'all_hists' in self.model_name:
                enc_delta_seq_out = delta_encoder(delta_input[:, 0, i, :])

                enc_pcount_out = pcount_encoder(res_pcount_input[:,0, i, :])

                enc_tbs_out = tbs_encoder(res_tbs_input[:, 0, i, :])

                if 'wqtenc' in self.model_name:
                    enc_qt_enc_out = qt_encoder(qtenc_input[:, 0, i, :])
                    step_encoddings.append(Flatten()(enc_qt_enc_out))
                    enc_last_acc_tb_out = last_acc_tb_encoder(last_acc_tb_input[:, 0, i, :])
                    step_encoddings.append(Flatten()(enc_last_acc_tb_out))


                step_encoddings.append(Flatten()(enc_delta_seq_out))
                step_encoddings.append(Flatten()(enc_pcount_out))
                step_encoddings.append(Flatten()(enc_tbs_out))

            
            else:
                enc_delta_seq_out = delta_encoder(tf.squeeze(delta_input, axis=1))

                enc_pcount_out = pcount_encoder(Flatten()(res_pcount_input))

                enc_tbs_out = tbs_encoder(Flatten()(res_tbs_input))

                if 'wqtenc' in self.model_name:
                    enc_last_acc_tb_out = last_acc_tb_encoder(Flatten()(last_acc_tb_input))


            if i == self.look_back-1:
                last_delta_seq = enc_delta_seq_out
                fled = Flatten()(enc_delta_seq_out)
                reped = RepeatVector(int(n_enc_unit/n_unit))(fled)
                final_ed = Reshape(target_shape=(1, n_enc_unit))(reped) 
                last_delta_seq_encod_length = final_ed

                flep = Flatten()(enc_pcount_out)
                repep = RepeatVector(int(n_unit/n_unit_l))(flep)
                final_ep = Reshape(target_shape=(1, n_unit))(repep) 
                last_pcount = final_ep

                flet = Flatten()(enc_tbs_out)
                repet = RepeatVector(int(n_unit/n_unit_l))(flet)
                final_et = Reshape(target_shape=(1, n_unit))(repet) 
                last_tbs = final_et

                if 'wqtenc' in self.model_name:

                    fleat = Flatten()(enc_last_acc_tb_out)
                    repeat = RepeatVector(int(n_unit/n_unit_l))(fleat)
                    finala_et = Reshape(target_shape=(1, n_unit))(repeat) 
                    lastacc_tb = finala_et

                    if 'all_hists' in self.model_name:
                        fleqt = Flatten()(enc_qt_enc_out)
                    else:
                        fleqt = Flatten()(qtenc_input)
                    repeqt = RepeatVector(int(n_unit/n_unit_l))(fleqt)
                    finalq_et = Reshape(target_shape=(1, n_unit))(repeqt) 
                    lastqt_enc = finalq_et
            
            if 'all_hists' in self.model_name:
                enc_out = tf.keras.layers.Concatenate(axis=-1)(step_encoddings)
            encoded_matrices.append(enc_out)

        lstm_input = tf.stack(encoded_matrices, axis=1)
        if 'all_hists' in self.model_name:
            combined_input = lstm_input
        else:
            if 'wqtenc' in self.model_name:
                encoded_feature_tensors = [lstm_input, lastqt_enc, lastacc_tb, last_delta_seq_encod_length, last_pcount, last_tbs]
            else:
                encoded_feature_tensors = [lstm_input, last_delta_seq_encod_length, last_pcount, last_tbs]

            combined_input = tf.keras.layers.Concatenate(axis=1)(encoded_feature_tensors)

        if 'wSelfAttention' in self.model_name:
            transformed_feature_tensors = tf.stack(encoded_feature_tensors, axis=1) #when lookback != 1, the first element is  (batchSize, lookback, n_unit) and does not match with the rest
            attention_scores = Dense(1, activation='tanh')(transformed_feature_tensors)
            attention_scores = tf.squeeze(attention_scores, axis=-1)
            attention_weights = Softmax(axis=1, name='attention_weights_layer')(attention_scores)
            attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1) 
            weighted_feature_tensors = transformed_feature_tensors * attention_weights_expanded
            combined_input = tf.reduce_sum(weighted_feature_tensors, axis=1)


        
        if 'MLPpred' in self.model_name:
            x1 = Dense(4 * n_unit, activation='relu', name='mlp_dense1')(combined_input)
            # x1 = Dropout(0.25)(x1)
            x2 = Dense(2 * n_unit, activation='relu', name='mlp_dense2')(x1)
            x2 = Dense(n_unit, activation='relu', name='mlp_dense3')(x1)
            x = Dropout(0.25)(x2)
        elif 'CNNpred' in self.model_name:
            x = Conv1D(filters=4*n_unit, kernel_size=1, activation='relu')(combined_input)
            x = Conv1D(filters=2*n_unit, kernel_size=1, activation='relu')(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(n_unit, activation='relu', name='cnn_dense3')(x)
            x = Reshape(target_shape=(1, n_unit), name='cnn_reshape')(x) 
            x = Dropout(0.25)(x)
        elif 'multiLSTM' in self.model_name:
            x = LSTM(4*n_unit, return_sequences=True, name='enc_lstm_2o')(combined_input)
            x, last_h, last_c = LSTM(2*n_unit, return_state=True, name='enc_lstm_2')(combined_input)
            x = Dropout(0.25)(x)
            x = RepeatVector(1, name='repeat_1')(x)
            x = LSTM(2*n_unit, return_sequences=True, name='dec_lstm_1')(x, initial_state=[ last_h, last_c])
            x = LSTM(n_unit, return_sequences=True, name='dec_lstm_1o')(x)
            x = Dropout(0.25)(x)
        else:
            x, last_h, last_c = LSTM(n_unit, return_state=True, name='enc_lstm_2')(combined_input)
            # x = Dropout(0.25)(x)
            x = RepeatVector(1, name='repeat_1')(x)
            repeat_vector_h = RepeatVector(1, name='repeat_2')(last_h)
            repeat_vector_h = tf.keras.backend.reshape(repeat_vector_h, (-1, n_unit))
            repeat_vector_c = RepeatVector(1, name='repeat_3')(last_c)
            repeat_vector_c = tf.keras.backend.reshape(repeat_vector_c, (-1, n_unit))
            x = LSTM(n_unit, return_sequences=True, name='dec_lstm_1')(x, initial_state=[repeat_vector_h, repeat_vector_c])


        delta_dense_output = []
        if self.output_type == 'consec_delta_simple':
            delta_out = Dense(units=self.pred_count_limit, name='dense_2')(x)
        if '_par_based_delta' in self.output_type:
            if 'wqtenc' in self.model_name:
                x1 = tf.keras.layers.Concatenate(axis=1)([x, lastqt_enc, lastacc_tb, last_delta_seq, last_tbs])
            else:
                x1 = tf.keras.layers.Concatenate(axis=1)([x, last_delta_seq, last_tbs])
            x1 = Flatten(name='xflat_delta')(x1)
            stacked_delta_output = Dense(self.delta_class_count, activation='sigmoid', name='mat_delta_dense')(x1)

        delta_output = Flatten(name='delta')(stacked_delta_output)
        
        x2 = tf.keras.layers.Concatenate(axis=1)([last_pcount, last_tbs])
        x2 = Flatten(name='xflat_pcount')(x2)
        pcount_out = Dense(units=self.pred_count_limit, activation='softmax', name='pcount_dense_out')(x2)
        pcount_output = Flatten(name='pcount')(pcount_out)

        x3 = tf.keras.layers.Concatenate(axis=1)([x, last_tbs])
        x3 = Flatten(name='xflat_tbs')(x3)
        res_tbs_out = Dense(units=len(Config.table_list), activation='sigmoid', name='res_tbs_dense_out')(x3)
        res_tbs_output = Flatten(name='tbs')(res_tbs_out)

        if 'wqtenc' in self.model_name:
            self.model = Model(inputs=[model_input, delta_input, res_tbs_input, res_pcount_input, qtenc_input, last_acc_tb_input], outputs=[delta_output, res_tbs_output, pcount_output])
        else:
            self.model = Model(inputs=[model_input, delta_input, res_tbs_input, res_pcount_input], outputs=[delta_output, res_tbs_output, pcount_output])

    def compile_and_fit(self, patience=7, restore=False):
        # Split data into training and validation sets
        validation_start_index = int(len(self.data_x) * 0.9)

        def split_data(data):
            return data[:validation_start_index], data[validation_start_index:]

        training_input_x, validation_input_x = split_data(self.data_x)
        training_input_x_delta, validation_input_x_delta = split_data(self.data_x_delta)
        training_input_x_tbs, validation_input_x_tbs = split_data(self.data_x_res_tbs)
        training_input_x_pcount, validation_input_x_pcount = split_data(self.data_x_res_pcount)

        if 'wqtenc' in self.model_name:
            training_input_x_qtenc, validation_input_x_qtenc = split_data(self.data_x_qt_enc)
            training_input_x_lastacctb, validation_input_x_lastacctb = split_data(self.data_x_last_tb)

        data_y = np.array(self.data_y)
        training_output, validation_output = split_data(data_y)
        training_output_tbs, validation_output_tbs = split_data(self.data_y_res_tbs)
        training_output_pcount, validation_output_pcount = split_data(self.data_y_res_pcount)

        # Determine learning rate
        if '_lowlr' in self.model_name:
            lr = 0.0001
        elif '_highllr' in self.model_name:
            lr = 0.01
        else:
            lr = 0.001

        # Define loss functions
        acb = '_acb_' in self.model_name
        if 'focal' in self.model_name:
            losses = {
                'delta': keras.losses.BinaryFocalCrossentropy(from_logits=False, apply_class_balancing=acb, alpha=ALPHA, gamma=GAMMA),
                'tbs': keras.losses.BinaryFocalCrossentropy(from_logits=False, apply_class_balancing=acb, alpha=ALPHA, gamma=GAMMA),
                'pcount': keras.losses.BinaryFocalCrossentropy(from_logits=False, apply_class_balancing=acb, alpha=ALPHA, gamma=GAMMA),
            }
        else:
            losses = {
                'delta': keras.losses.BinaryCrossentropy(from_logits=False),
                'tbs': keras.losses.BinaryCrossentropy(from_logits=False),
                'pcount': keras.losses.BinaryCrossentropy(from_logits=False),
            }

        # Compile the model
        if self.output_type == 'consec_delta_simple':
            self.model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()]
            )
        elif '_par_based_delta' in self.output_type and 'ftune' not in self.model_name:
            self.model.compile(
                loss=losses,
                loss_weights={'delta': 1, 'tbs': 1, 'pcount': 0.8},
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                metrics=['accuracy']
            )
        else:
            self.model.compile(
                loss=losses,
                loss_weights={'delta': 1, 'tbs': 0.5, 'pcount': 0.5},
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                metrics=['accuracy']
            )

        # Set training configurations
        monitor = 'val_delta_accuracy' if 'valAcc' in self.model_name else 'val_delta_loss'
        mode = 'max' if 'valAcc' in self.model_name else 'min'
        batch_size = 128 if 'b128' in self.model_name else 64

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=restore
        )

        # Prepare data generators or direct training
        if 'noVal' in self.model_name:
            history = self.model.fit(
                [training_input_x, training_input_x_delta, training_input_x_tbs, training_input_x_pcount],
                [training_output, training_output_tbs, training_output_pcount],
                epochs=EPOCH_COUNT,
                verbose=1,
                validation_data=(
                    [validation_input_x, validation_input_x_delta, validation_input_x_tbs, validation_input_x_pcount],
                    [validation_output, validation_output_tbs, validation_output_pcount]
                )
            )
        elif 'wqtenc' in self.model_name:
            training_generator = DataGenerator(
                [training_input_x, training_input_x_delta, training_input_x_tbs, training_input_x_pcount, training_input_x_qtenc, training_input_x_lastacctb],
                [training_output, training_output_tbs, training_output_pcount],
                batch_size=batch_size
            )
            history = self.model.fit(
                training_generator,
                epochs=EPOCH_COUNT,
                verbose=1,
                validation_data=(
                    [validation_input_x, validation_input_x_delta, validation_input_x_tbs, validation_input_x_pcount, validation_input_x_qtenc, validation_input_x_lastacctb],
                    [validation_output, validation_output_tbs, validation_output_pcount]
                ),
                callbacks=[early_stopping],
                batch_size=batch_size
            )
        else:
            training_generator = DataGenerator(
                [training_input_x, training_input_x_delta, training_input_x_tbs, training_input_x_pcount],
                [training_output, training_output_tbs, training_output_pcount],
                batch_size=batch_size
            )
            history = self.model.fit(
                training_generator,
                epochs=EPOCH_COUNT,
                verbose=1,
                validation_data=(
                    [validation_input_x, validation_input_x_delta, validation_input_x_tbs, validation_input_x_pcount],
                    [validation_output, validation_output_tbs, validation_output_pcount]
                ),
                callbacks=[early_stopping],
                batch_size=batch_size
            )

        # Store the model and return the training history
        store_model(self.model, self.model_name)
        return history

    def test_model(self):
        global SKIPPED_PREFETCH_STEPS
        Config.tb_bid_range = get_tables_bid_range()
        Config.actual_tb_bid_range = get_tables_actual_bid_range()
        if qenc_time_test:
            with open(f'./Times/{Config.db_name}/qenc.txt', 'a') as ofile:
                ofile.write(f'{self.model_name}\n')
        global NON_EXISTING_DELTAS, SKIPPED_PREFETCH_STEPS
        sep = Config.csv_file_separator
        test_time = 0
        test_prep_time = 0
        tests = get_dataset_workloads(Config.db_name, False, False, measure_time)
        tests_outputs = {} 

        if measure_time and Config.db_name in ['auctionl1v2', 'tpcc1v2', 'wiki_sf100_2']:
            db_helper.create_clone_for_transactional_test()
            conn = db_helper.get_db_connection()
            pretest_tests = get_db_pretest_queries(Config.db_name)
            for test in pretest_tests:
                file_name = f'{Config.db_name}_{test}{self.config_suffix}WB{Config.logical_block_size}WP{Config.max_partition_size}'
                test_log_lines: List[LogLine] = get_log_lines(file_name, False, get_queries=measure_time, test_name=test)
                for i in tqdm(range(len(test_log_lines)), desc=f'Pretest run {test}'):
                    db_helper.get_query_execution_time(test_log_lines[i].query.statement, conn)
            Config.tb_bid_range = get_tables_bid_range(False)
            Config.actual_tb_bid_range = get_tables_actual_bid_range(False)


        if 'deltaMappingLin' in self.res_file_name:
            self.get_train_q_delta_clusters()
            self.mapping_functions = None

        for k in k_option:
            true_k = k
            print(f'k: {k}- rs: {res_multiplier}')

            if '_regulateK' in self.res_file_name:
                k = int(k * 128 / Config.max_partition_size)

            Config.prefetching_k = k
            tlim = None
            if Config.db_name == 'tpcds':
                tlim = 50  
            ti = -1
            method_res = {}
            for test_idx in range(len(tests)):
                if IS_ANALYZING_TIME:
                    tlim = None
                    if Config.db_name == 'tpcds':
                        tlim = 50 

                if test_idx > 0 and 'deltaMappingLin' in self.res_file_name:
                    self.get_delta_mappings_from_test(tests[:test_idx])

                test = tests[test_idx]
                file_name = f'{Config.db_name}_{test}{self.config_suffix}WB{Config.logical_block_size}WP{Config.max_partition_size}'
                if test not in tests_outputs:
                    test_df = pd.read_csv(
                        "./Data/" + file_name + ".txt",
                        header=0,
                        sep=sep,
                        quoting=csv.QUOTE_ALL,
                        quotechar='"',
                        engine='python',
                    )
                    if tlim: test_df = test_df.iloc[:tlim]
                    if 'ClientIP' in test_df.columns:
                        test_df.rename(columns={'ClientIP': 'clientIP'}, inplace=True)  

                    if Config.db_name in ['genomic', 'birds']:  
                        test_df['qPlan'] = '-'              
                    if (Config.db_name not in ['genomic', 'birds']) and ('qPlan' not in test_df.columns or '-' in test_df['qPlan'].to_list()):  
                        test_df['qPlan'] = '-'
                        conn = db_helper.get_db_connection()
                        cursor = conn.cursor()
                    
                        print(test_df.head())

                        for idx, row in test_df.iterrows():
                            stmt = row['statement'].lower()
                            explain_query = f'EXPLAIN (COSTS off) {stmt}'
                            cursor.execute(explain_query)
                            plan_rows = cursor.fetchall()
                            plan_text = ' '.join(row[0] for row in plan_rows if 'Workers Planned' not in row[0])
                            test_df.loc[idx, 'qPlan'] = plan_text.replace(sep, ' ')
                        print(test_df.head())
                        my_to_csv(test_df, f'./Data/{file_name}.txt', sep=sep)


                    if ('qJsonPlan'not in test_df.columns or '-' in test_df['qJsonPlan'].to_list() or IS_ANALYZING_TIME) and 'qjsn' in self.model_name:  
                        print(f'Getting the query json plans. size of df before {len(test_df)}')
                        test_df['qJsonPlan'] = '-'
                        conn = db_helper.get_db_connection()
                        cursor = conn.cursor()
                    
                        t1 = time.time()
                        for idx, row in test_df.iterrows():
                            stmt = row['statement'].lower()
                            explain_query = f'EXPLAIN (COSTS off, FORMAT JSON) {stmt}'
                            cursor.execute(explain_query)
                            plan_rows = cursor.fetchall()
                            test_df.loc[idx, 'qJsonPlan'] = json.dumps(plan_rows[0][0])
                        t2 = time.time()
                        test_prep_time += t2 - t1

                        if not IS_ANALYZING_TIME:
                            my_to_csv(test_df, f'./Data/{file_name}.txt', sep=sep)
                        print(f'size of new df after {len(test_df)}')
                            
                    
                    test_df['clientIP'] = '127.0.0.1'
                    test_df['encResultBlock'] = '-'
                    test_df['partitionNumbers_perTb'] = '-'
                    test_df['partitionNumbers'] = test_df['resultPartitions'].apply(lambda x: sorted([int(re.search(r'\d+', item).group()) for item in x.split(',')]))
                    test_df = get_complete_query_result_details3(test_df, self.partition_manager, pid_selection_mode=self.pid_selection_mode)

                    if 'wqtenc' in self.model_name:
                        if qenc_time_test:
                            rand_idx = random.randint(0, len(test_df)-1)
                            test_df = test_df[rand_idx: rand_idx+1]

                        if 'qjsn' in self.model_name:
                            test_df['processedJPlan'] = '-'
                            for idx, row in test_df.iterrows():
                                t1 = time.time()
                                stmt = row['statement'].lower()
                                jplan = json.loads(row['qJsonPlan'])
                                q_type_repr, tables, join_conditions, filter_conditions = self.encode_json_plan(stmt, jplan)

                                alias_refs = {}                    
                                for tb, alias in tables:
                                    alias_refs[alias] = tb

                                join_conditions_per_tb = {}
                                for ac_idx, cond in enumerate(join_conditions):
                                    for alias in alias_refs:
                                        if alias in cond:
                                            tb = alias_refs[alias]
                                            curr_cond = join_conditions_per_tb.get(tb, '')
                                            join_conditions_per_tb[tb] = f'{curr_cond} {cond}'
                                
                                filter_conditions_per_tb = {}
                                for ac_idx, (alias, cond) in enumerate(filter_conditions):
                                    tb = alias_refs[alias]
                                    curr_cond = filter_conditions_per_tb.get(tb, '')
                                    filter_conditions_per_tb[tb] = f'{curr_cond} {cond}'
                                
                                t2 = time.time()
                                t1_2 = t2 - t1
                                test_df.loc[idx, 'processedJPlan'] = json.dumps([q_type_repr, list(alias_refs.values()), join_conditions_per_tb, filter_conditions_per_tb, stmt])

                            t3 = time.time()
                            test_df['qt_enc'] = test_df['processedJPlan'].apply(self.get_query_representation)
                            t4 = time.time()
                            t3_4 = t4 - t3
                            if qenc_time_test:
                                with open(f'./Times/{Config.db_name}/qenc.txt', 'a') as ofile:
                                    ofile.write(f'{test}: \t process={round(t1_2*1000, 4)}, encode={round(t3_4*1000, 4)}, total={round((t1_2+t3_4)*1000, 4)}\n')
                            test_prep_time += t4 - t1
                        else:
                            if self.query_template_encoding_method == 'sql':
                                t1 = time.time()
                                test_df['prep_stmt'] = test_df['statement'].apply(process_plan_text)
                                t2 = time.time()
                            else:
                                t1 = time.time()
                                test_df['prep_stmt'] = test_df['qPlan'].apply(process_plan_text)
                                t2 = time.time()
                            t3 = time.time()
                            test_df['prep_stmt'] = test_df['prep_stmt'].apply(preprocess_plan_text)
                            test_df['qt_enc'] = test_df['prep_stmt'].apply(self.get_query_template_enc)
                            t4 = time.time()
                            if qenc_time_test:
                                with open(f'./Times/{Config.db_name}/qenc.txt', 'a') as ofile:
                                    ofile.write(f'{test}: \t process={round((t2-t1)*1000, 4)}, encode={round((t4-t3)*1000, 4)}, total={round((t2-t1 + t4-t3)*1000, 4)}\n')

                    if qenc_time_test:
                        continue
                    encoded_seqs = []
                    deltas_seqs = []
                    res_tbs_seqs = []
                    res_pcount_seqs = []
                    qt_enc_seqs = []
                    last_tb_seqs = []
                    deltas_seq = []
                    enc_seq = []
                    res_pcount_seq = []
                    res_tbs_seq = []
                    test_res_per_tb = []
                    qt_enc_seq = []
                    last_tb_seq = []

                    t1 = time.time()

                    for j in range(0, Config.look_back - 1):
                        enc_seq.append(np.zeros((len(Config.table_list), Config.encoding_length)))
                        # enc_seq.append(np.zeros((len(Config.table_list), 2 * Config.encoding_length)))
                        deltas_seq.append({})
                        res_pcount_seq.append(np.zeros(self.pred_count_limit))
                        res_tbs_seq.append(np.zeros(len(Config.table_list)))
                        if 'wqtenc' in self.model_name:
                            last_tb_seq.append(np.zeros(len(Config.table_list)))
                            qt_enc_seq.append(np.zeros(self.qt_encod_size))
                    enc_seq.append(get_encoded_block_aggregation(test_df.loc[0, 'encResultBlock'], False))
                    deltas_seq.append(self.get_par_deltas_per_table(None, test_df.loc[0, 'partitionNumbers_perTb']))
                    res_pcount_seq.append([1 if ii == min(test_df.loc[0, 'resultParCount'], self.pred_count_limit) else 0 for ii in range(1, self.pred_count_limit+1)])
                    res_tbs_seq.append(get_encoded_res_tbs(test_df.loc[0, 'resultTables']))
                    if 'wqtenc' in self.model_name:
                        qt_enc_seq.append(test_df.loc[0, 'qt_enc'])
                        last_tb_seq.append([0 if tbt != self.get_last_acc_tb(test_df.loc[0, 'resultTables']) else 1 for tbt in Config.table_list])
                    for i in range(1, len(test_df)):
                        test_res_per_tb.append(self.get_query_tb_based_res(test_df.loc[i, 'partitionNumbers_perTb']))
                        if test_df.loc[i, 'encResultBlock'] == '-':
                            continue
                        if test_df.loc[i, 'clientIP'] == test_df.loc[i - 1, 'clientIP']:
                            enc_seq.append(get_encoded_block_aggregation(test_df.loc[i, 'encResultBlock'], False))
                            q_delta = self.get_par_deltas_per_table(test_df.loc[i-1, 'partitionNumbers_perTb'], test_df.loc[i, 'partitionNumbers_perTb'])
                            deltas_seq.append(q_delta)
                            res_tbs_seq.append(get_encoded_res_tbs(test_df.loc[i, 'resultTables']))
                            if 'wqtenc' in self.model_name:
                                qt_enc_seq.append(test_df.loc[i, 'qt_enc'])
                                last_tb_seq.append([0 if tbt != self.get_last_acc_tb(test_df.loc[i, 'resultTables']) else 1 for tbt in Config.table_list])
                            res_pcount_seq.append([1 if ii == min(test_df.loc[i, 'resultParCount'], self.pred_count_limit) else 0 for ii in range(1, self.pred_count_limit+1)])

                        elif len(enc_seq) > 0:
                            encoded_seqs.append(enc_seq)
                            deltas_seqs.append(deltas_seq)
                            res_tbs_seqs.append(res_tbs_seq)
                            res_pcount_seqs.append(res_pcount_seq)
                            if 'wqtenc' in self.model_name:
                                qt_enc_seqs.append(qt_enc_seq)
                                last_tb_seqs.append(last_tb_seq)
                            enc_seq = [get_encoded_block_aggregation(test_df.loc[i, 'encResultBlock'], False)]
                            q_delta = self.get_par_deltas_per_table(None, test_df.loc[i, 'partitionNumbers_perTb'])
                            deltas_seq = [q_delta]
                            res_tbs_seq[get_encoded_res_tbs(test_df.loc[i, 'resultTables'])]
                            res_pcount_seq[[1 if ii == min(test_df.loc[i, 'resultParCount'], self.pred_count_limit) else 0 for ii in range(1, self.pred_count_limit+1)]]
                            if 'wqtenc' in self.model_name:
                                qt_enc_seq = [test_df.loc[i, 'qt_enc']]
                                last_tb_seq = [[0 if tbt != self.get_last_acc_tb(test_df.loc[i, 'resultTables']) else 1 for tbt in Config.table_list]]
                    if len(enc_seq) > 0:
                        encoded_seqs.append(enc_seq)
                        deltas_seqs.append(deltas_seq)
                        res_tbs_seqs.append(res_tbs_seq)
                        res_pcount_seqs.append(res_pcount_seq)
                        if 'wqtenc' in self.model_name:
                            qt_enc_seqs.append(qt_enc_seq)
                            last_tb_seqs.append(last_tb_seq)


                    # Convert the sequences to actual input/output for the model
                    test_data_x, test_data_y = [], []
                    test_data_x_delta = []
                    test_data_x_tbs, test_data_x_pcount = [], []
                    test_data_x_qt_enc, test_data_x_last_tb = [], []
                    test_data_y_tbs, test_data_y_pcount = [], []
                    for j in range(len(encoded_seqs)):
                        # for j in range(5):
                        for i in range(len(encoded_seqs[j]) - self.look_back):
                            test_data_x.append(np.array(encoded_seqs[j][i:(i + self.look_back)]))

                            if 'all_hists' in self.model_name:
                                hist_deltas = []
                                for ih in range(self.look_back):
                                    r_deltas = []
                                    step_delta = deltas_seqs[j][i+ih]
                                    if 'biDelta' in self.model_name:
                                        normalized_delta = self.get_binary_delta_class(step_delta, True)
                                    else:
                                        for tb in Config.table_list:
                                            if tb in step_delta:
                                                normalized_delta = (np.array(step_delta[tb]) - self.deltas_mean) / self.deltas_std
                                                if self.max_p_count_of_tbs[tb] - len(normalized_delta) > 0:
                                                    normalized_delta = np.concatenate([normalized_delta, np.zeros(self.max_p_count_of_tbs[tb] - len(normalized_delta))])
                                                elif len(normalized_delta) > self.max_p_count_of_tbs[tb]:
                                                    normalized_delta = normalized_delta[:self.max_p_count_of_tbs[tb]]
                                            else:
                                                normalized_delta = np.zeros(self.max_p_count_of_tbs[tb])
                                            
                                            r_deltas.append(normalized_delta)
                                        normalized_delta = np.concatenate(r_deltas)
                                    hist_deltas.append(normalized_delta)

                                test_data_x_delta.append(np.array(hist_deltas))
                                test_data_x_tbs.append(np.array(res_tbs_seqs[j][i:(i + self.look_back)]))
                                test_data_x_pcount.append(np.array(res_pcount_seqs[j][i:(i + self.look_back)]))
                                if 'wqtenc' in self.model_name:
                                    test_data_x_qt_enc.append(np.array(qt_enc_seqs[j][i:(i + self.look_back)]))
                                    test_data_x_last_tb.append(np.array(last_tb_seqs[j][i:(i + self.look_back)]))
                            else:
                                r_deltas = []
                                step_delta = deltas_seqs[j][i+self.look_back-1]
                                if 'biDelta' in self.model_name:
                                    normalized_delta = self.get_binary_delta_class(step_delta, True)
                                else:
                                    for tb in Config.table_list:
                                        if tb in step_delta:
                                            normalized_delta = (np.array(step_delta[tb]) - self.deltas_mean) / self.deltas_std
                                            if self.max_p_count_of_tbs[tb] - len(normalized_delta) > 0:
                                                normalized_delta = np.concatenate([normalized_delta, np.zeros(self.max_p_count_of_tbs[tb] - len(normalized_delta))])
                                            elif len(normalized_delta) > self.max_p_count_of_tbs[tb]:
                                                normalized_delta = normalized_delta[:self.max_p_count_of_tbs[tb]]
                                        else:
                                            normalized_delta = np.zeros(self.max_p_count_of_tbs[tb])       
                                        r_deltas.append(normalized_delta)

                                normalized_delta = np.concatenate(r_deltas)

                                test_data_x_delta.append(normalized_delta)
                                test_data_x_tbs.append(res_tbs_seqs[j][i+self.look_back-1])
                                test_data_x_pcount.append(res_pcount_seqs[j][i+self.look_back-1])
                                if 'wqtenc' in self.model_name:
                                    if '_mixenc' in self.model_name:
                                        test_data_x_qt_enc.append(np.array(qt_enc_seqs[j][i:(i+self.look_back)]))
                                    else:
                                        test_data_x_qt_enc.append(np.array(qt_enc_seqs[j][i+self.look_back-1]))
                                    test_data_x_last_tb.append(np.array(last_tb_seqs[j][i+self.look_back-1]))

                            if self.output_type == 'consec_delta_simple':
                                normalized_delta_y = np.array(deltas_seqs[j][i+self.look_back])
                                if len(normalized_delta_y) > self.pred_count_limit:
                                    normalized_delta_y = normalized_delta_y[:self.pred_count_limit]
                                elif len(normalized_delta_y) < self.pred_count_limit:
                                    normalized_delta_y = np.concatenate([normalized_delta_y, np.zeros(self.pred_count_limit - len(normalized_delta_y))])
                            if '_par_based_delta' in self.output_type:
                                normalized_delta_y = self.get_binary_delta_class(deltas_seqs[j][i+self.look_back], False)
                                normalized_delta_y = np.array(normalized_delta_y)
                            test_data_y.append(normalized_delta_y)
                            test_data_y_tbs.append(res_tbs_seqs[j][i+self.look_back])
                            test_data_y_pcount.append(res_pcount_seqs[j][i+self.look_back])

                    # Make the model input and output
                    t2 = time.time()
                    test_prep_time += t2 - t1
                    
                    print(f'\n\t\t\t #Missing_deltas in test = {NON_EXISTING_DELTAS}')
                    NON_EXISTING_DELTAS = 0
                    test_data_x = np.array(test_data_x)
                    test_data_x_delta = np.expand_dims(np.array(test_data_x_delta), axis=1)
                    test_data_x_tbs = np.expand_dims(np.array(test_data_x_tbs), axis=1)
                    test_data_x_pcount = np.expand_dims(np.array(test_data_x_pcount), axis=1)
                    if 'wqtenc' in self.model_name:
                        test_data_x_qt_enc = np.expand_dims(np.array(test_data_x_qt_enc), axis=1)
                        test_data_x_last_tb = np.expand_dims(np.array(test_data_x_last_tb), axis=1)
                    test_data_y = np.array(test_data_y)
                    test_data_y_tbs = np.array(test_data_y_tbs)
                    test_data_y_pcount = np.array(test_data_y_pcount)
                    test_data_x = test_data_x.reshape(test_data_x.shape[0], self.look_back, test_data_x.shape[2] * test_data_x.shape[3])
                    print(test_data_x.shape)
                    print(test_data_x_delta.shape)
                    print(test_data_y.shape)

                    t1 = time.time()

                    if 'wqtenc' in self.model_name:
                        output = self.model.predict([test_data_x, test_data_x_delta, test_data_x_tbs, test_data_x_pcount, test_data_x_qt_enc, test_data_x_last_tb], verbose=1)
                    else:
                        output = self.model.predict([test_data_x, test_data_x_delta, test_data_x_tbs, test_data_x_pcount], verbose=1)
                    
                    if 'wqtenc' in self.model_name:
                        tests_outputs[test] = [output, test_res_per_tb, qt_enc_seq]
                    else:
                        tests_outputs[test] = [output, test_res_per_tb]
                    t2 = time.time()
                    prediction_time = t2 - t1
                    print(len(output[0]))
                    if IS_ANALYZING_TIME:
                        with open(f'{Config.result_base_path}/logFiles/{Config.db_name}_log_data.txt', 'a') as outfi:
                            outfi.write(f'\nmodel_{self.model_name} avg prediction time for {len(test_df)} queries: {prediction_time / len(test_df)}')
                            outfi.write(f'\nmodel_{self.model_name} avg test preparation time for {len(test_df)} queries: {test_prep_time / len(test_df)}')
                            outfi.write(f'\nmodel_{self.model_name} avg total prediction time for {len(test_df)} queries: {(prediction_time + test_prep_time) / len(test_df)}')
                else:
                    if 'wqtenc' in self.model_name:
                        output, test_res_per_tb, qt_enc_seq = tests_outputs[test]
                    else:
                        output, test_res_per_tb = tests_outputs[test]
                    print(len(output[0]))


                # Test begin
                test_log_lines: List[LogLine] = get_log_lines(file_name, False, get_queries=measure_time, test_name=test)
                if measure_time and 'tpcc' in Config.db_name:
                    tlim = 500
                
                if IS_ANALYZING_TIME:
                    tlim = int(max(len(test_log_lines) // 10, 10))

                if tlim: test_log_lines = test_log_lines[:tlim]
                pref_par_cache = LRUCache(200)
                par_all_cache = LRUCache(self.cache_size//Config.max_partition_size)
                pref_block_cache = LRUCache(16000)
                general_cache = LRUCache(self.cache_size)
                par_general_cache = LRUCache(self.cache_size)
                all_cache = LRUCache(self.cache_size)
                last_prefetch = LRUCache(self.cache_size)
                last_prefetch_par = LRUCache(self.cache_size // Config.max_partition_size)
                step_sole_hits_and_misses = []
                step_sole_hits_and_misses_par = []
                res_hit_miss = []
                first_block_retrival = 0
                sum_q_exec_time = 0
                sum_pref_time = 0
                requested_blocks = set()
                prefetched_blocks = set()
                for tr in range(test_repeat):

                    SKIPPED_PREFETCH_STEPS = 0
                    tb_select_thresh = TB_SELECT_THRESHOLD



                    all_ever_cached = set()

                    requested_pars = set()
                    prefetched_pars = set()

                    res_size_error = []
                    correct_par_args = []
                    correct_res_size_rank = []
                    pred_val_diff_true = []
                    correct_acc_tbs_preds = []
                    min_correct_acc_tbs_preds = []
                    max_tb_prob_errs = []

                    fpref_time = 0
            
                    print(f'{test}-{tr}')
                    if measure_time: db_helper.clear_cache()
                    for i in tqdm(range(len(test_log_lines))):

                        if measure_time:
                            db_helper.clear_sys_cache()
                            conn = db_helper.get_db_connection()
                            q_exec_time = db_helper.get_query_execution_time(test_log_lines[i].query.statement, conn)
                            sum_q_exec_time += q_exec_time


                        if tr == 0:
                            h_hit = all_cache.hit_count
                            h_miss = all_cache.miss_count
                            np_contents = []
                            if measure_cache_stats:
                                np_contents=general_cache.get_all_content()
                            for b_id in test_log_lines[i].query.result_set:
                                requested_blocks.add(b_id)
                                general_cache.put(b_id, increase_hit=True, np_contents=np_contents, check_insertions=measure_cache_stats)
                                all_cache.put(b_id, increase_hit=True, np_contents=np_contents, check_insertions=measure_cache_stats)
                                last_prefetch.put(b_id, increase_hit=True, np_contents=np_contents, check_insertions=measure_cache_stats)

                            requested_partitions = test_log_lines[i].query.result_partitions
                            for p_id in requested_partitions:
                                requested_pars.add(p_id)
                                par_all_cache.put(p_id, increase_hit=True)
                                last_prefetch_par.put(p_id, increase_hit=True)

                            q_hit = all_cache.hit_count - h_hit
                            q_miss = all_cache.miss_count - h_miss
                            res_hit_miss.append([q_hit, q_miss])
                            step_sole_hits_and_misses.append([last_prefetch.hit_count, last_prefetch.miss_count, last_prefetch.total_pres])
                            last_prefetch.clear()
                            step_sole_hits_and_misses_par.append([last_prefetch_par.hit_count, last_prefetch_par.miss_count, last_prefetch_par.total_pres])
                            last_prefetch_par.clear()
                            self.partition_manager.update_partition_graph(requested_partitions)

                        if i == len(test_log_lines) - 1:
                            continue

                        if measure_cache_stats:
                            all_cache.increase_evicted_keys_counter()
                            par_all_cache.increase_evicted_keys_counter()

                        prediction = output[0][i]

                        ft1 = time.time()
                        arg_sorted_res_size_pred = np.argsort(output[2][i])
                        true_res_size = min(len(test_log_lines[i+1].query.result_partitions), self.pred_count_limit)
                        top_index = arg_sorted_res_size_pred[-1] + 1
                        res_size_error.append(true_res_size - top_index)
                        correct_res_size_rank.append(arg_sorted_res_size_pred.tolist().index(true_res_size-1))
                        res_lst = output[2][i].tolist()
                        pred_val_diff_true.append(res_lst[top_index-1] - res_lst[true_res_size-1])
                        top_index *= res_multiplier
                        
                        true_acc_tbs = get_encoded_res_tbs(test_log_lines[i+1].query.get_query_accessed_tables())
                        true_acc_tbs = np.array(true_acc_tbs)
                        correct_tbs_probs = output[1][i][true_acc_tbs == 1]
                        min_tb_prob = min(correct_tbs_probs)
                        min_correct_acc_tbs_preds.append(min_tb_prob)
                        correct_acc_tbs_preds.extend(correct_tbs_probs.tolist())
                        
                        thresholded_pred = np.where(output[1][i] > tb_select_thresh, 1, 0)
                        
                        indices_of_ones = np.where(thresholded_pred == 1)[0]
                        if dynamic_tb_thresh:
                            if min_tb_prob < tb_select_thresh:
                                new_tb_select_thresh = tb_select_thresh - tb_alpha * max(1, (len(correct_tbs_probs) - len(indices_of_ones)))#(tb_select_thresh - min_tb_prob)/1.5
                                tb_select_thresh = max(min(min_correct_acc_tbs_preds), new_tb_select_thresh)
                            elif min_tb_prob > tb_select_thresh:
                                new_tb_select_thresh = tb_select_thresh + tb_alpha/10
                                tb_select_thresh = min(statistics.mean(min_correct_acc_tbs_preds), new_tb_select_thresh)
                            
                        if len(indices_of_ones) == 0:
                            indices_of_ones = [np.argmax(output[1][i])]
                        next_res_tbs = []
                        for tb_idx in indices_of_ones:
                            next_res_tbs.append(Config.table_lookup[tb_idx+1].lower())


                        if 'wqtenc' in self.model_name:
                            partitions_to_prefetch = self.find_par_to_prefetch(prediction, test_res_per_tb[i], k_lim=min(k, top_index), acc_tb_idx=indices_of_ones, qt_enc=qt_enc_seq[i], tbs=next_res_tbs)
                        else:
                            partitions_to_prefetch = self.find_par_to_prefetch(prediction, test_res_per_tb[i], k_lim=min(k, top_index), acc_tb_idx=indices_of_ones, qt_enc=None, tbs=next_res_tbs)

                        prefetched_partitions = [str(partition) for partition in partitions_to_prefetch[:k]]
                            
                                
                        ft2 = time.time()
                        fpref_time += ft2 - ft1
                        blocks_to_insert = []
                        step_pred_pars = [] #keeps rank of correct predictions
                        for pred_i in range(len((prefetched_partitions))):
                            pred_par = prefetched_partitions[pred_i]
                            if pred_par in test_log_lines[i+1].query.result_partitions:
                                step_pred_pars.append(pred_i)
                            p_block_list = self.partition_manager.partitions.get(pred_par).blocks
                            
                            if tr == 0:
                                par_all_cache.put(pred_par, increase_hit=False, insert_type='p')
                                last_prefetch_par.put(pred_par, increase_hit=False, insert_type='p')

                                np_contents = []
                                if measure_cache_stats: np_contents=general_cache.get_all_content()
                                
                                self.partition_manager.put_partition_in_cache(pred_par, all_cache, allowed_tbs=next_res_tbs, np_contents=np_contents, check_insertions=measure_cache_stats)
                                self.partition_manager.put_partition_in_cache(pred_par, last_prefetch, allowed_tbs=next_res_tbs, np_contents=np_contents, check_insertions=measure_cache_stats)

                                for b_id in p_block_list:
                                    prefetched_blocks.add(b_id)
                                prefetched_pars.add(pred_par)
                            if measure_time:
                                for b_id in p_block_list:
                                    blocks_to_insert.append(b_id)
                        
                        if len(step_pred_pars) == 0:
                            step_pred_pars.append(self.pred_count_limit + 1)
                        correct_par_args.append(max(step_pred_pars))

                        t1 = time.time()
                        if measure_time:
                            pref_time = db_helper.insert_blocks_to_cache(blocks_to_insert)
                            sum_pref_time += pref_time
                        t2 = time.time()
                        fpref_time += t2 - t1

                    if total_repeat == 1:
                        print('------------------------------------')
                        print(sum_pref_time)
                        print(sum_q_exec_time)
                        print(f'Skipped prefetching {SKIPPED_PREFETCH_STEPS} times.')
                        print(pref_par_cache.report('Partition Cache'))
                        print(pref_block_cache.report('Block Cache'))
                        print(general_cache.report('General Cache'))
                        print(all_cache.get_full_report('All Cache'))
                        print(par_all_cache.get_full_report('All Cache Partition'))
                        print(f'final tb thresh = {tb_select_thresh}')
                        p, r = get_acc_prec_recall(step_sole_hits_and_misses)
                        print(f'Block level avg precision = {p}, avg recall = {r}')
                        p, r = get_acc_prec_recall(step_sole_hits_and_misses_par)
                        print(f'Partition level avg precision = {p}, avg recall = {r}')

                        print('------------------------------------')
                    
                    res_dict = {
                        'block_cache': pref_block_cache.report_dict(),
                        'partition_cache': pref_par_cache.report_dict(),
                        'combined_partition_cache': par_all_cache.report_dict(),
                        'general_block_cache': general_cache.report_dict(),
                        'combined_block_cache': all_cache.report_dict(),
                        'res_hit_miss': res_hit_miss,
                        'res_sole_hit_miss': step_sole_hits_and_misses,
                        'res_sole_hit_miss_par': step_sole_hits_and_misses_par,
                    }

                    if tr == 0:
                        useless = 0
                        for b_id in prefetched_blocks:
                            if b_id not in requested_blocks:
                                useless += 1

                        par_useless = 0
                        for pid in prefetched_pars:
                            if pid not in requested_pars:
                                par_useless += 1

                        res_dict['skipped_prefetches'] = SKIPPED_PREFETCH_STEPS

                        res_dict['useless_prefs'] = int(useless)
                        res_dict['useful_prefs'] = len(prefetched_blocks) - int(useless)
                        print(f"block pref precision = {res_dict['useful_prefs']/len(prefetched_blocks)}")
                        res_dict['block_access_count'] = general_cache.get_size()
                        res_dict['partition_access_count'] = par_general_cache.get_size()

                        res_dict['useless_par_prefs'] = int(par_useless)
                        res_dict['useful_par_prefs'] = len(prefetched_pars) - int(par_useless)
                        print(f"par pref precision = {res_dict['useful_par_prefs']/len(prefetched_pars)}")
                        res_dict['total_misses'] = general_cache.get_total_access() - general_cache.hit_count
                        res_dict['eliminated_misses'] = res_dict['total_misses'] - (
                                    all_cache.get_total_access() - all_cache.hit_count)
                        res_dict['eliminated_misses_par'] = par_general_cache.get_total_access() - par_general_cache.hit_count - (
                                    par_all_cache.get_total_access() - par_all_cache.hit_count)

                        for key in res_dict.keys():
                            method_res[f'{test}_{key}'] = res_dict[key]
                            # method_res[f'{test}_pred_time'] = pred_time

                    if tr == test_repeat - 1:
                        method_res[f'{test}_exec_time'] = sum_q_exec_time / test_repeat
                        method_res[f'{test}_pref_time'] = sum_pref_time / test_repeat

            iteration = 0
            if save_to_file:
                print('saving')        
                if '_regulateK' in self.res_file_name:
                    Config.prefetching_k = true_k   
                ending = '_timed' if measure_time else ''
                full_res_file_name = f'{result_base_path}/{Config.db_name}_{self.res_file_name}_{Config.prefetching_k * Config.max_partition_size}_WB{Config.logical_block_size}WP{Config.max_partition_size}{ending}.p'
                if iteration > 0:
                    avg_k = 1 if iteration < total_repeat - 1 else total_repeat
                    try:
                        res_history = cPickle.load(open(full_res_file_name, 'rb'))
                        method_res = aggregate_result_dicts(res_history, method_res, avg_k)
                    except FileNotFoundError:
                        print(f'!!! File {full_res_file_name} did not exist !!!')
    
                cPickle.dump(method_res, open(full_res_file_name, 'wb'))
                print(full_res_file_name)

        print('done')

    def find_par_to_prefetch(self, prediction, result_partitions, k_lim, acc_tb_idx, qt_enc=None, tbs=[]):
        global SKIPPED_PREFETCH_STEPS, SKIP_PREFETCH_STEP
        if len(tbs) == 0:
            print('nothing to select')
            return []
        top_k_indices = np.argsort(prediction)[-k_lim:]
        if top_k_indices[0] == 0 and SKIP_PREFETCH_STEP:
            SKIPPED_PREFETCH_STEPS += 1
            return []
        if 'first' in self.output_type:
            last_p = result_partitions[0]
        elif 'mid' in self.output_type:
            last_p = result_partitions[len(result_partitions) // 2]
        else:
            last_p = result_partitions[-1]

        last_p_tb = last_p.rsplit('_', 1)[0]
        last_p = int(last_p.rsplit('_', 1)[1])
        candid_pars = []
        invalid_count = 0

        for idx in top_k_indices:
            if idx == 0:
                continue
            scale = 1
            try:
                delta = self.delta_classes_reverse[idx]                
            except Exception as e:
                if 'DelC' in self.res_file_name:
                    continue
                else:
                    raise Exception
            delta = int(delta)
            for tb in tbs: 
                if delta not in self.freq_tb_based_deltas[tb]:
                    continue
                
                candid_par = last_p + int(delta * scale)
                if candid_par < 0: continue
                if f'{tb}_{candid_par}' not in self.partition_manager.partitions:
                    invalid_count += 1
                    continue
                if f'{tb}_{candid_par}' not in candid_pars:
                    candid_pars.append(f'{tb}_{candid_par}')
        return candid_pars

    def load_model_from_file(self):
        all_info_dict = cPickle.load(open(f"{base_model_file_dir}{self.model_name}{FQTESTSUFFIX}_all_info_dict.p", 'rb'))
        self.delta_classes = all_info_dict['delta_classes']
        self.delta_classes_reverse = all_info_dict['delta_classes_reverse']
        self.max_p_count = all_info_dict['max_p_count']
        self.max_p_count_of_tbs = all_info_dict['max_p_count_of_tbs']
        self.deltas_mean = all_info_dict['deltas_mean']
        self.deltas_std = all_info_dict['deltas_std']
        if "delta_class_count" in all_info_dict:
            self.delta_class_count = all_info_dict["delta_class_count"]
        if "pred_count_limit" in all_info_dict:
            self.pred_count_limit = all_info_dict["pred_count_limit"]
        if 'grasp' in self.model_name:
            self.freq_tb_based_deltas = all_info_dict["freq_tb_based_deltas"]
        self.model = load_model(self.model_name)
        if '_wqtenc_' in self.model_name and 'qjsn_simple' not in self.model_name:
            if 'qjsn' in self.model_name:
                qtenc_model_file_name = f'./SavedFiles/Models/qt_encoders/{self.file_name}_{self.query_template_encoding_method}_qtj_preprocL{PLAN_PREP_LEVEL}.model'
            else:
                qtenc_model_file_name = f'./SavedFiles/Models/qt_encoders/{self.file_name}_{self.query_template_encoding_method}_preprocL{PLAN_PREP_LEVEL}.model'
            try:
                self.qt_encoder = Doc2Vec.load(qtenc_model_file_name)
            except FileNotFoundError:
                if Config.db_name in ['birds', 'genomic']:
                    pass
                else:
                    qtenc_model_file_name = qtenc_model_file_name.replace(f'_preprocL{PLAN_PREP_LEVEL}', '')
                    self.qt_encoder = Doc2Vec.load(qtenc_model_file_name)
        
    def check_model_exist(self):
        try:
            load_model(self.model_name)
        except Exception as e:
            print(f'Model: {self.model_name} does not exist')
            err = traceback.format_exc()
            print(err)
            return False
        return True

    def plot_delta_analysis(self, lims=[8000]):
        db_mode = [['auctionl1v1', 1], ['auctionl1v1', 2], ['auction_sf25_1', 2]]

        plt.rcParams.update({
            'axes.titlesize': 15.2,  
            'axes.labelsize': 15,    
            'xtick.labelsize': 12.5, 
            'ytick.labelsize': 12    
        })

        while True:
            lim = int(input("enter lim\t"))
            x = float(input("enter the x size"))
            y = float(input("enter the y size"))

            dbs_data = []
        
            for db, mode in db_mode:
                if mode == 2:
                    db_data = cPickle.load(open(f'{db}_data_x_delta_stb_delta__WP{Config.max_partition_size}{self.output_type}.p', 'rb'))
                else:
                    db_data = cPickle.load(open(f'{db}_data_x_delta_adr__WP{Config.max_partition_size}_{self.output_type}.p', 'rb'))
                
                dbs_data.append(db_data)
            
            fig, axs = plt.subplots(3, 1, figsize=(x, y * 3)) 

            colors = plt.cm.get_cmap('tab10', len(Config.table_list))

            for plot_idx in range(3): 
                ax = axs[plot_idx] 
                if db_mode[plot_idx][1] == 1:
                    x_vals = []
                    y_vals = []
                    data_x_delta = dbs_data[plot_idx]

                    for i, inner_list in enumerate(data_x_delta[:lim]):
                        for value in inner_list:
                            if value != 0:
                                x_vals.append(i)  
                                y_vals.append(value)
                    
                    ax.scatter(x_vals, y_vals, s=0.69)
                else:
                    alter_config(db_mode[plot_idx][0], Config.max_partition_size)
                    data_x_delta = dbs_data[plot_idx]
                    x_vals = {}
                    y_vals = {}

                    for tb in Config.table_list:
                        y_vals[tb] = []
                        x_vals[tb] = []

                    for i, inner_dict in enumerate(data_x_delta[:lim]):
                        for tb in inner_dict:
                            for delta in inner_dict[tb]:
                                if delta != 0:
                                    x_vals[tb].append(i)  
                                    y_vals[tb].append(delta) 

                    for idx, tb in enumerate(Config.table_list):
                        ax.scatter(x_vals[tb], y_vals[tb], s=0.75, color=colors(idx), label=tb)
                
                if plot_idx == 0:
                    ax.set_ylabel('Delta Values', labelpad=-1)
                    ax.set_title(f'(a)', loc='right', fontweight='bold')
                    ax.set_yticks([-1000, 0, 1000])

                elif plot_idx == 1:
                    ax.set_ylabel('Delta Values', labelpad=0)
                    ax.set_title(f'(b)', loc='right', fontweight='bold')
                    ax.set_yticks([-400, 0, 400])
                elif plot_idx == 2:
                    ax.set_ylabel('Delta Values', labelpad=0)
                    ax.set_title(f'(c)', loc='right', fontweight='bold')

                if plot_idx == 2:
                    ax.set_xlabel('Query Number')
                else:
                    ax.set_xticklabels([])

                
            plt.tight_layout() 
            plt.subplots_adjust(hspace=0.34)
            plt.savefig(f'./Figures/deltas/group_delta_vis_lim{lim}_{Config.db_name}_{self.config_suffix}_{self.output_type}WP{Config.max_partition_size}.pdf', bbox_inches='tight')

            plt.show()
            
            quit_command = input("q to stop")
            if quit_command == 'q':
                break

    def process_plan_node(self, plan, tables, join_conditions, filter_conditions):
        # Extract table and alias
        if "Relation Name" in plan:
            relation_name = plan["Relation Name"]
            alias = plan.get("Alias", relation_name)  # Use alias if available, otherwise the table name
            tables.append([relation_name, alias])

        # Extract join condition
        if "Hash Cond" in plan:
            join_conditions.append(plan["Hash Cond"])
        if "Join Filter" in plan:
            join_conditions.append(plan["Join Filter"])

        # Extract index or filter condition
        if "Index Cond" in plan:
            if 'Alias' not in plan and "Relation Name" not in plan:
                pass
            else:
                filter_conditions.append([plan["Alias"], plan["Index Cond"]])
        if "Filter" in plan:
            if 'Alias' not in plan and "Relation Name" not in plan:
                pass
            else:
                filter_conditions.append([plan["Alias"], plan["Filter"]])
        
        if "Recheck Cond" in plan:
            filter_conditions.append([plan["Alias"], plan["Recheck Cond"]])


        # Recursively process nested plans
        if "Plans" in plan:
            for subplan in plan["Plans"]:
                self.process_plan_node(subplan, tables, join_conditions, filter_conditions)

    def encode_json_plan(self, sql, plan):
        tables = []
        join_conditions = []
        filter_conditions = []
        q_type_repr = [0] * 4#[0, 0, 0, 0] # Delete, Insert, Select, Update (alphabetically)
        for node in plan:
            if node['Plan'].get('Operation') == 'Insert':
                q_type_repr[1] = 1
                tables.append([node['Plan']['Relation Name'], node['Plan']['Relation Name']])
                filter_conditions.append([tables[0][0], sql])
                return q_type_repr, tables, join_conditions, filter_conditions
            elif  node['Plan'].get('Operation') == 'Update':
                q_type_repr[3] = 1
            elif  node['Plan'].get('Operation') == 'Delete':
                q_type_repr[0] = 1
            else:
                q_type_repr[2] = 1

            self.process_plan_node(node['Plan'], tables, join_conditions, filter_conditions)
        if 'qjsn_simple' in self.model_name:
            return q_type_repr, tables, [], []
        if 'qjsn_q2v' in self.model_name:
            return q_type_repr, tables, sql
        return q_type_repr, tables, join_conditions, filter_conditions

    def get_query_representation(self, q_comp_j):
        q_comp = json.loads(q_comp_j)
        q_type = q_comp[0]
        tables = [0 for _ in range(len(Config.table_list))]
        join_conds = [[0 for __ in range(8)] for _ in range(len(Config.table_list))]
        filter_conds = [[0 for __ in range(8)] for _ in range(len(Config.table_list))]
        for tb_idx, tb in enumerate(Config.db_name):
            if tb in q_comp[1]:
                tables[tb_idx] = 1

            if 'qjsn_simple' in self.model_name:
                continue
        
            if tb in q_comp[2]:
                cond_enc = self.qt_encoder.infer_vector(preprocess_plan_text(q_comp[2][tb]))
                join_conds[tb_idx] = cond_enc
            if tb in q_comp[3]:
                cond_enc = self.qt_encoder.infer_vector(preprocess_plan_text(q_comp[3][tb]))
                filter_conds[tb_idx] = cond_enc
        
        if 'qjsn_nlp' in self.model_name:
            stmt_enc = self.qt_encoder.infer_vector(preprocess_plan_text(q_comp[4]))
            final_repr = q_type + tables + stmt_enc.tolist()
            remaining = self.qt_encod_size - len(final_repr)
            final_repr += [0 for i in range(remaining)]
        elif 'qjsn_simple' in self.model_name:
                return  q_type + tables
        elif 'qjsn_mask' in self.model_name:
            return [0 for i in range(self.qt_encod_size)]
        else:
            final_repr = q_type + tables + [item for sublist in join_conds for item in sublist] + [item for sublist in filter_conds for item in sublist]
        return final_repr     

    def set_db_scales(self, test_scale, train_scale):
        self.train_db_scale = int(train_scale)
        self.test_db_scale = int(test_scale)

    def fine_tune(self, config_suffix, ftune_db):
        try:
            global EPOCH_COUNT, ALPHA, GAMMA
            EPOCH_COUNT = FTUNE_EPOCHS
            prev_model_name = self.model_name
            self.model_name = prev_model_name.replace(config_suffix, f'{config_suffix}_ftune_fq{FTUNE_QUERY_COUNT}_fe{FTUNE_EPOCHS}')
            if 'tpch' in Config.db_name:
                self.model_name = prev_model_name.replace(config_suffix, f'{config_suffix}_ft')
                raise FileNotFoundError
            if IS_ANALYZING_TIME:
                raise FileNotFoundError
            self.model = load_model(self.model_name)
            self.model_name = prev_model_name

            # self.load_model_from_file()
        except FileNotFoundError:
            err = traceback.format_exc()
            print(err)
            print('loading ftuned model failed, training it.')
            sep = Config.csv_file_separator
            file_name = f'{ftune_db}_all_train{config_suffix}WB{Config.logical_block_size}WP{Config.max_partition_size}'
            test_df = pd.read_csv(
                "./Data/" + file_name + ".txt",
                header=0,
                sep=sep,
                quoting=csv.QUOTE_ALL,
                quotechar='"',
                engine='python',
            )
            tlim = FTUNE_QUERY_COUNT
            if 'tpcc' in Config.db_name:
                df_value1 = test_df[test_df['clientIP'] == '10.20.30.40']
                df_value2 = test_df[test_df['clientIP'] == '10.20.30.41']
                df_value1_trimmed = df_value1.iloc[-tlim//2:]
                df_value2_trimmed = df_value2.iloc[-tlim//2:] 
                test_df = pd.concat([df_value1_trimmed, df_value2_trimmed])
                test_df = test_df.reset_index(drop=True)

            else:
                test_df = test_df.iloc[-tlim:].reset_index(drop=True)
            print(len(test_df))
            if 'ClientIP' in test_df.columns:
                test_df.rename(columns={'ClientIP': 'clientIP'}, inplace=True)  

            test_df['clientIP'] = '127.0.0.1'
            test_df['encResultBlock'] = '-'
            test_df['partitionNumbers_perTb'] = '-'
            test_df['partitionNumbers'] = test_df['resultPartitions'].apply(lambda x: sorted([int(re.search(r'\d+', item).group()) for item in x.split(',')]))
            test_df = get_complete_query_result_details3(test_df, self.partition_manager, pid_selection_mode=self.pid_selection_mode)

            if 'wqtenc' in self.model_name:
                if 'qjsn' in self.model_name:
                    test_df['processedJPlan'] = '-'
                    for idx, row in test_df.iterrows():
                        stmt = row['statement'].lower()
                        jplan = json.loads(row['qJsonPlan'])
                        q_type_repr, tables, join_conditions, filter_conditions = self.encode_json_plan(stmt, jplan)

                        alias_refs = {}                    
                        for tb, alias in tables:
                            alias_refs[alias] = tb

                        join_conditions_per_tb = {}
                        for ac_idx, cond in enumerate(join_conditions):
                            for alias in alias_refs:
                                if alias in cond:
                                    tb = alias_refs[alias]
                                    curr_cond = join_conditions_per_tb.get(tb, '')
                                    join_conditions_per_tb[tb] = f'{curr_cond} {cond}'
                        
                        filter_conditions_per_tb = {}
                        for ac_idx, (alias, cond) in enumerate(filter_conditions):
                            tb = alias_refs[alias]
                            curr_cond = filter_conditions_per_tb.get(tb, '')
                            filter_conditions_per_tb[tb] = f'{curr_cond} {cond}'
                        
                        test_df.loc[idx, 'processedJPlan'] = json.dumps([q_type_repr, list(alias_refs.values()), join_conditions_per_tb, filter_conditions_per_tb, stmt])

                    test_df['qt_enc'] = test_df['processedJPlan'].apply(self.get_query_representation)
                else:
                    if self.query_template_encoding_method == 'sql':
                        test_df['prep_stmt'] = test_df['statement'].apply(process_plan_text)
                    else:
                        test_df['prep_stmt'] = test_df['qPlan'].apply(process_plan_text)
                    test_df['prep_stmt'] = test_df['prep_stmt'].apply(preprocess_plan_text)
                    test_df['qt_enc'] = test_df['prep_stmt'].apply(self.get_query_template_enc)

            encoded_seqs = []
            deltas_seqs = []
            res_tbs_seqs = []
            res_pcount_seqs = []
            qt_enc_seqs = []
            last_tb_seqs = []
            deltas_seq = []
            enc_seq = []
            res_pcount_seq = []
            res_tbs_seq = []
            test_res_per_tb = []
            qt_enc_seq = []
            last_tb_seq = []

            self.delta_class_count_limit = len(self.delta_classes)

            t1 = time.time()

            for j in range(0, Config.look_back - 1):
                enc_seq.append(np.zeros((len(Config.table_list), Config.encoding_length)))
                # enc_seq.append(np.zeros((len(Config.table_list), 2 * Config.encoding_length)))
                deltas_seq.append({})
                res_pcount_seq.append(np.zeros(self.pred_count_limit))
                res_tbs_seq.append(np.zeros(len(Config.table_list)))
                if 'wqtenc' in self.model_name:
                    last_tb_seq.append(np.zeros(len(Config.table_list)))
                    qt_enc_seq.append(np.zeros(self.qt_encod_size))
            enc_seq.append(get_encoded_block_aggregation(test_df.loc[0, 'encResultBlock'], False))
            deltas_seq.append(self.get_par_deltas_per_table(None, test_df.loc[0, 'partitionNumbers_perTb']))
            res_pcount_seq.append([1 if ii == min(test_df.loc[0, 'resultParCount'], self.pred_count_limit) else 0 for ii in range(1, self.pred_count_limit+1)])
            res_tbs_seq.append(get_encoded_res_tbs(test_df.loc[0, 'resultTables']))
            if 'wqtenc' in self.model_name:
                qt_enc_seq.append(test_df.loc[0, 'qt_enc'])
                last_tb_seq.append([0 if tbt != self.get_last_acc_tb(test_df.loc[0, 'resultTables']) else 1 for tbt in Config.table_list])
            for i in range(1, len(test_df)):
                test_res_per_tb.append(self.get_query_tb_based_res(test_df.loc[i, 'partitionNumbers_perTb']))
                if test_df.loc[i, 'encResultBlock'] == '-':
                    continue
                if test_df.loc[i, 'clientIP'] == test_df.loc[i - 1, 'clientIP']:
                    enc_seq.append(get_encoded_block_aggregation(test_df.loc[i, 'encResultBlock'], False))
                    q_delta = self.get_par_deltas_per_table(test_df.loc[i-1, 'partitionNumbers_perTb'], test_df.loc[i, 'partitionNumbers_perTb'])
                    deltas_seq.append(q_delta)
                    res_tbs_seq.append(get_encoded_res_tbs(test_df.loc[i, 'resultTables']))
                    if 'wqtenc' in self.model_name:
                        qt_enc_seq.append(test_df.loc[i, 'qt_enc'])
                        last_tb_seq.append([0 if tbt != self.get_last_acc_tb(test_df.loc[i, 'resultTables']) else 1 for tbt in Config.table_list])
                    res_pcount_seq.append([1 if ii == min(test_df.loc[i, 'resultParCount'], self.pred_count_limit) else 0 for ii in range(1, self.pred_count_limit+1)])

                elif len(enc_seq) > 0:
                    encoded_seqs.append(enc_seq)
                    deltas_seqs.append(deltas_seq)
                    res_tbs_seqs.append(res_tbs_seq)
                    res_pcount_seqs.append(res_pcount_seq)
                    if 'wqtenc' in self.model_name:
                        qt_enc_seqs.append(qt_enc_seq)
                        last_tb_seqs.append(last_tb_seq)
                    enc_seq = [get_encoded_block_aggregation(test_df.loc[i, 'encResultBlock'], False)]
                    q_delta = self.get_par_deltas_per_table(None, test_df.loc[i, 'partitionNumbers_perTb'])
                    deltas_seq = [q_delta]
                    res_tbs_seq[get_encoded_res_tbs(test_df.loc[i, 'resultTables'])]
                    res_pcount_seq[[1 if ii == min(test_df.loc[i, 'resultParCount'], self.pred_count_limit) else 0 for ii in range(1, self.pred_count_limit+1)]]
                    if 'wqtenc' in self.model_name:
                        qt_enc_seq = [test_df.loc[i, 'qt_enc']]
                        last_tb_seq = [[0 if tbt != self.get_last_acc_tb(test_df.loc[i, 'resultTables']) else 1 for tbt in Config.table_list]]
            if len(enc_seq) > 0:
                encoded_seqs.append(enc_seq)
                deltas_seqs.append(deltas_seq)
                res_tbs_seqs.append(res_tbs_seq)
                res_pcount_seqs.append(res_pcount_seq)
                if 'wqtenc' in self.model_name:
                    qt_enc_seqs.append(qt_enc_seq)
                    last_tb_seqs.append(last_tb_seq)


            # Convert the sequences to actual input/output for the model
            test_data_x, test_data_y = [], []
            test_data_x_delta = []
            test_data_x_tbs, test_data_x_pcount = [], []
            test_data_x_qt_enc, test_data_x_last_tb = [], []
            test_data_y_tbs, test_data_y_pcount = [], []
            for j in range(len(encoded_seqs)):
                # for j in range(5):
                for i in range(len(encoded_seqs[j]) - self.look_back):
                    test_data_x.append(np.array(encoded_seqs[j][i:(i + self.look_back)]))

                    if 'all_hists' in self.model_name:
                        hist_deltas = []
                        for ih in range(self.look_back):
                            r_deltas = []
                            step_delta = deltas_seqs[j][i+ih]
                            if 'biDelta' in self.model_name:
                                normalized_delta = self.get_binary_delta_class(step_delta)
                            else:
                                for tb in Config.table_list:
                                    if tb in step_delta:
                                        normalized_delta = (np.array(step_delta[tb]) - self.deltas_mean) / self.deltas_std
                                        if self.max_p_count_of_tbs[tb] - len(normalized_delta) > 0:
                                            normalized_delta = np.concatenate([normalized_delta, np.zeros(self.max_p_count_of_tbs[tb] - len(normalized_delta))])
                                        elif len(normalized_delta) > self.max_p_count_of_tbs[tb]:
                                            normalized_delta = normalized_delta[:self.max_p_count_of_tbs[tb]]
                                    else:
                                        normalized_delta = np.zeros(self.max_p_count_of_tbs[tb])
                                    
                                    r_deltas.append(normalized_delta)
                                normalized_delta = np.concatenate(r_deltas)
                            hist_deltas.append(normalized_delta)

                        test_data_x_delta.append(np.array(hist_deltas))
                        test_data_x_tbs.append(np.array(res_tbs_seqs[j][i:(i + self.look_back)]))
                        test_data_x_pcount.append(np.array(res_pcount_seqs[j][i:(i + self.look_back)]))
                        if 'wqtenc' in self.model_name:
                            test_data_x_qt_enc.append(np.array(qt_enc_seqs[j][i:(i + self.look_back)]))
                            test_data_x_last_tb.append(np.array(last_tb_seqs[j][i:(i + self.look_back)]))
                    else:
                        r_deltas = []
                        step_delta = deltas_seqs[j][i+self.look_back-1]
                        if 'biDelta' in self.model_name:
                            normalized_delta = self.get_binary_delta_class(step_delta)
                        else:
                            for tb in Config.table_list:
                                if tb in step_delta:
                                    normalized_delta = (np.array(step_delta[tb]) - self.deltas_mean) / self.deltas_std
                                    if self.max_p_count_of_tbs[tb] - len(normalized_delta) > 0:
                                        normalized_delta = np.concatenate([normalized_delta, np.zeros(self.max_p_count_of_tbs[tb] - len(normalized_delta))])
                                    elif len(normalized_delta) > self.max_p_count_of_tbs[tb]:
                                        normalized_delta = normalized_delta[:self.max_p_count_of_tbs[tb]]
                                else:
                                    normalized_delta = np.zeros(self.max_p_count_of_tbs[tb])       
                                r_deltas.append(normalized_delta)

                        normalized_delta = np.concatenate(r_deltas)

                        test_data_x_delta.append(normalized_delta)
                        test_data_x_tbs.append(res_tbs_seqs[j][i+self.look_back-1])
                        test_data_x_pcount.append(res_pcount_seqs[j][i+self.look_back-1])
                        if 'wqtenc' in self.model_name:
                            if '_mixenc' in self.model_name:
                                test_data_x_qt_enc.append(np.array(qt_enc_seqs[j][i:(i+self.look_back)]))
                            else:
                                test_data_x_qt_enc.append(np.array(qt_enc_seqs[j][i+self.look_back-1]))
                            test_data_x_last_tb.append(np.array(last_tb_seqs[j][i+self.look_back-1]))

                    if self.output_type == 'consec_delta_simple':
                        normalized_delta_y = np.array(deltas_seqs[j][i+self.look_back])
                        if len(normalized_delta_y) > self.pred_count_limit:
                            normalized_delta_y = normalized_delta_y[:self.pred_count_limit]
                        elif len(normalized_delta_y) < self.pred_count_limit:
                            normalized_delta_y = np.concatenate([normalized_delta_y, np.zeros(self.pred_count_limit - len(normalized_delta_y))])
                    if '_par_based_delta' in self.output_type:
                        normalized_delta_y = self.get_binary_delta_class(deltas_seqs[j][i+self.look_back])
                        normalized_delta_y = np.array(normalized_delta_y)
                    test_data_y.append(normalized_delta_y)
                    test_data_y_tbs.append(res_tbs_seqs[j][i+self.look_back])
                    test_data_y_pcount.append(res_pcount_seqs[j][i+self.look_back])

            # Make the model input and output
            test_data_x = np.array(test_data_x)
            test_data_x_delta = np.expand_dims(np.array(test_data_x_delta), axis=1)
            test_data_x_tbs = np.expand_dims(np.array(test_data_x_tbs), axis=1)
            test_data_x_pcount = np.expand_dims(np.array(test_data_x_pcount), axis=1)
            if 'wqtenc' in self.model_name:
                test_data_x_qt_enc = np.expand_dims(np.array(test_data_x_qt_enc), axis=1)
                test_data_x_last_tb = np.expand_dims(np.array(test_data_x_last_tb), axis=1)
            test_data_y = np.array(test_data_y)
            test_data_y_tbs = np.array(test_data_y_tbs)
            test_data_y_pcount = np.array(test_data_y_pcount)
            test_data_x = test_data_x.reshape(test_data_x.shape[0], self.look_back, test_data_x.shape[2] * test_data_x.shape[3])
            print(test_data_x.shape)
            print(test_data_x_delta.shape)
            print(test_data_y.shape)
            print(test_data_x_tbs.shape)
            if 'wqtenc' in self.model_name:
                print(test_data_x_qt_enc.shape)
                print(test_data_x_last_tb.shape)
            print(test_data_x_pcount.shape)
            print(test_data_y_pcount.shape)
            

            for layer in self.model.layers:
                layer.trainable = False

            self.model.get_layer('pcount_dense_out').trainable = True
            self.model.get_layer('res_tbs_dense_out').trainable = True
            self.model.get_layer('mat_delta_dense').trainable = True 

            self.data_x_delta = test_data_x_delta
            self.data_y = np.array(test_data_y)
            self.data_x = test_data_x
            self.data_x_res_tbs = test_data_x_tbs
            self.data_x_qt_enc = test_data_x_qt_enc
            self.data_x_last_tb = test_data_x_last_tb
            self.data_x_res_pcount = test_data_x_pcount
            self.data_y_res_pcount = test_data_y_pcount
            self.data_y_res_tbs = test_data_y_tbs
            
            self.compile_and_fit(patience=5)
            t2 = time.time()
            self.model_name = prev_model_name
            if IS_ANALYZING_TIME:
                with open(f'{Config.result_base_path}/logFiles/{Config.db_name}_log_data.txt', 'a') as outfi:
                    outfi.write(f'\nmodel_{self.model_name} fine tuning time: {t2 - t1}')
        
    def load_new_classes(self, new_model_name):
        all_info_dict = cPickle.load(open(f"{base_model_file_dir}{new_model_name}{FQTESTSUFFIX}_all_info_dict.p", 'rb'))
        new_delta_classes = all_info_dict['delta_classes']
        new_delta_classes_reverse = all_info_dict['delta_classes_reverse']

        if '_loadDelC' in self.model_name:
            self.delta_classes = new_delta_classes
            self.delta_classes_reverse = new_delta_classes_reverse
            return

        elif '_addDelC'in self.model_name:
            new_deltas = set(new_delta_classes.keys())
            cur_deltas = set(self.delta_classes.keys())
            deltas_to_add = list(new_deltas.difference(cur_deltas))
            intact_deltas = new_deltas.intersection(cur_deltas)
            print(f'{len(deltas_to_add)} deltas to add and {len(intact_deltas)} are similar')
            max_used_delta_cls = max(self.delta_classes_reverse.keys()) + 1

            i = -1
            for i, delta in enumerate(deltas_to_add):
                self.delta_classes[delta] = max_used_delta_cls + i
                self.delta_classes_reverse[max_used_delta_cls + i] = delta
            
            assert i+1 == len(deltas_to_add)
        
        target_freq_tb_based_deltas = all_info_dict["freq_tb_based_deltas"]
        for tb in Config.table_list:
            if tb in target_freq_tb_based_deltas:
                curr_freq_tb_based_deltas = self.freq_tb_based_deltas.get(tb, [])
                curr_freq_tb_based_deltas.extend(target_freq_tb_based_deltas[tb])
                self.freq_tb_based_deltas[tb] = curr_freq_tb_based_deltas

def get_ftune_model_name(model_name, delta_db, dc_count, output_type='last_par_based_delta'):
    directory_path = './SavedFiles/Models'
    semantic_idx = model_name.index('grasp')
    model_name = model_name[semantic_idx:].replace('_addDelC', '').replace(Config.db_name, delta_db)
    if 'tpch_sf30' in delta_db:
        model_name = model_name.replace('NewMLP2', 'NewMLP2_ftune_frmSF10')
    return f'{delta_db}_{model_name}'
        
def get_encoded_res_tbs(res_tb_list):
    ertb = [0] * len(Config.table_list)
    for tb in Config.table_list:
        if tb in res_tb_list:
            ertb[Config.table_lookup[tb]-1] = 1
    return ertb

def aggregate_result_dicts(hist_dict, res_dict, k):
    result = {}   
    for key in res_dict:
        try:
            if isinstance(res_dict[key], dict) and isinstance(hist_dict[key], dict):
                result[key] = aggregate_result_dicts(res_dict[key], hist_dict[key], k)
            else:
                result[key] = (float(res_dict[key]) + float(hist_dict[key]))/k
        except Exception as e:
            print(f'!!Skipped key {key}')
            print(e.__repr__())
    return result

def prefix_counter(counter, prefix):
    return Counter({f"{prefix}_{key}": value for key, value in counter.items()})
  
def process_plan_text(plan_text):
    plan_text = plan_text.lower()
    plan_text = re.sub(r'::\w+', '', plan_text)

    plan_text = re.sub(r"'[^']*'", "VALUE", plan_text)
    plan_text = re.sub(r'\b\d+\b', 'N', plan_text)

    plan_text = re.sub(r'\s+', ' ', plan_text).strip()
    plan_text = re.sub(r' ->', '.', plan_text)

    return plan_text

def remove_unwanted_strs(plan_text):    
    if PLAN_PREP_LEVEL > 0:
        plan_text = re.sub(r'\([^)]*\)', '', plan_text)

        plan_text = re.sub(r'\b\d+\b', 'N', plan_text)
        plan_text = re.sub(r'\b\d+\.\d+\b', 'N', plan_text)

    if PLAN_PREP_LEVEL > 1:
        plan_text = re.sub(r'\s{2,}', ' ', plan_text)
        plan_text = re.sub(r'->', '', plan_text)
        plan_text = re.sub(r'::\w+', '', plan_text)
    
    return plan_text


def preprocess_plan_text(plan_text):
    plan_text = remove_unwanted_strs(plan_text)

    plan_text = plan_text.lower()
    tokens = word_tokenize(plan_text)
    tokens = [word for word in tokens if word.isalnum()]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

import json 

def get_acc_prec_recall(hits_n_misses):
    precisions = [h/c if c > 0 else 1 for h, m, c in hits_n_misses]
    recalls = [h/(h+m) if (h+m) > 0 else 0 for h, m, c in hits_n_misses]

    avg_precision = sum(precisions) / len(precisions)
    avg_recalls = sum(recalls) / len(recalls)

    return avg_precision, avg_recalls


import getopt
if __name__ == '__main__':
    NON_EXISTING_DELTAS = 0
    argumentList = sys.argv[1:]
    options = "d:s:c:p"
    long_options = ["Database=", "config=", "predModel=", "GPU=", "twice", "regulate", "epochs=", "focal", "tfocal", "acb", "focalConfig",
                    "alpha=", "gamma=", "jp", "dmapping=", "dmqst=", "binary_delta", "loadDeltaCls", "addDeltaCls", "dcCount=", 
                    "ftuneQC=", "ftuneEpch=", "grasp", "qjson", "planType=", "skip", "dynTBThresh", "rs=", "k=", "lookback=", 'time', 'pSize=',
                    'cSize=', 'testRepeat=', 'TBAlpha=', 'tau=', 'ngr', 'fqcountTest', 'fqcountTrain', 'timeAnalysis', 'second_db=']

    called_func = ''
    args = []

    received_args = {
        '--epochs' : 25,
        '--GPU' : "0",
        '--all_hists': True
    }

    try:
        arguments, values = getopt.getopt(argumentList, options, long_options)
        for currentArgument, currentValue in arguments:
            received_args[currentArgument] = currentValue
            if currentArgument == '--GPU':
                print(f'--GPU = {currentValue}')
    except getopt.error as err:
        print (str(err))
        exit()
 
    
    if  "--GPU" in received_args:
        os.environ["CUDA_VISIBLE_DEVICES"] = received_args['--GPU']

    k_option = [50]
    
    base_model_file_dir = "./SavedFiles/Models/grasp/"
    result_base_path = './Results'
    pref_log_file_base = f'{result_base_path}/logFiles/'
    
    config_suffix_ = NewMLPft_sequential_tbPID
    
    ### Run Flow ###
    just_plot = False
    do_load_model = False   
    check_model_existence = False
    if '--jp' in received_args:
        just_plot = True

    IS_ANALYZING_TIME = '--timeAnalysis' in received_args

    ### Model Conifgs ###
    block_level = False
    use_plan = False
    if '--planType' in received_args:
        use_plan = True
    use_mixed_enc_for_q = False
    use_lstm_encoding_for_qenc = False
    regulate_train_size = False
    pred_model = 'multiLSTM' #'_MLPpred' # '' is enc-dec lstm model
    output_types = ['first_par_based_delta']# ['mid_par_based_delta', 'last_par_based_delta', 'first_par_based_delta']
    Config.look_back = int(received_args.get('--lookback',2))
    EPOCH_COUNT = int(received_args['--epochs'])
    model_suffix = ''
    if EPOCH_COUNT != 75:
        model_suffix = f'epc{EPOCH_COUNT}'
    use_all_hists = '--all_hists' in received_args
    use_binary_delta = '--binary_delta' in received_args
    FTUNE_QUERY_COUNT = 5000
    FTUNE_EPOCHS = 20
    if '--ftuneQC' in received_args:
        FTUNE_QUERY_COUNT = int(received_args['--ftuneQC'])
    if '--ftuneEpch' in received_args:
        FTUNE_EPOCHS = int(received_args['--ftuneEpch'])
    
    PLAN_PREP_LEVEL = 1 #PPL
    FQTESTSUFFIX = ''
    if '--fqcountTest' in received_args:
        FQTESTSUFFIX = f'p{FTUNE_QUERY_COUNT}'

    QLIMIT = -1
    if '--fqcountTrain' in received_args:
        FQTESTSUFFIX = f'p{FTUNE_QUERY_COUNT}'
        QLIMIT = FTUNE_QUERY_COUNT


    ### Test Configs ###
    save_to_file = True
    measure_cache_stats = False
    measure_res_pred_stats = True
    measure_time = '--time' in received_args
    total_repeat = 1
    test_repeat = int(received_args.get('--testRepeat', 1))

    generalization_test = True
    if '--ngr' in received_args:
        generalization_test = False
    ignore_tb_prediction = False
    res_multiplier_opts = [50]
    par_sizes = [32]
    cache_GBs = [8]
    cache_GB = cache_GBs[0]
    cache_size = 16500 * cache_GB
    SKIPPED_PREFETCH_STEPS = 0
    SKIP_PREFETCH_STEP = True if '--skip' in received_args else False
    if SKIP_PREFETCH_STEP:
        print('SKIP is ON!')
    qenc_time_test = False
    dynamic_tb_thresh = '--dynTBThresh' in received_args
    rs = [int(received_args.get('--rs', 50))]
    tb_alpha = float(received_args.get('--TBAlpha', 0.05))
    TB_SELECT_THRESHOLD = float(received_args.get('--tau', 0.2))

    ###### Handling Deltas ######
    load_new_delta_classes = False
    add_new_delta_classes = False
    if '--loadDeltaCls' in received_args:
        load_new_delta_classes = True
        add_new_delta_classes = False
    if '--addDeltaCls' in received_args:
        load_new_delta_classes = False
        add_new_delta_classes = True
    scale_deltas = False
    save_small_deltas = False
    do_delta_mapping = False
    if '--dmapping' in received_args:
        do_delta_mapping = True
    DELTA_MAPPING_QTENC_SIM_THRESH = 0.5
    if '--dmqst' in received_args:
        DELTA_MAPPING_QTENC_SIM_THRESH = float(received_args['--dmqst'])
    tb_based_scale_deltas = False

    
    if "--regulate" in received_args:
        regulate_train_size = True
        use_mixed_enc_for_q = False
    twice_class_size = False
    if "--twice" in received_args:
        twice_class_size = True
        use_mixed_enc_for_q = False

    if '--predModel' in received_args:
        pred_model = received_args['--predModel']

    done_db = ''

    db_names = ['sdss_1']

    if  "--Database" in received_args:
        db_names = [received_args['--Database'].replace(',', '')]
        alter_config(db_names[0])
        if db_names[0] in ['sdss_4', 'wiki_sf100_1', 'auctionl1v1', 'tpcc1v1']:
            load_new_delta_classes = False
            add_new_delta_classes = False
    
    if '--second_db' in received_args:
        second_db = received_args['--second_db']
        

    block_level_name_suffix = '_blvl' if block_level else '' 
    res_combo_suffix = ''
    if ignore_tb_prediction:
        res_combo_suffix += '_alltbs'
    if scale_deltas:
        res_combo_suffix += '_linscale'
    if tb_based_scale_deltas:
        res_combo_suffix += '_tbBasedLinscale'
    if save_small_deltas and (scale_deltas or tb_based_scale_deltas):
        res_combo_suffix += '_saveSml10'

    if do_delta_mapping:
        if '--dmapping' in received_args:
            if len(received_args['--dmapping']) == 1:
                res_combo_suffix += f"_deltaMappingLin{received_args['--dmapping']}{str(DELTA_MAPPING_QTENC_SIM_THRESH).replace('.', '')}"
            else:
                res_combo_suffix += f"_{received_args['--dmapping']}deltaMappingLin{str(DELTA_MAPPING_QTENC_SIM_THRESH).replace('.', '')}"
        else:
            res_combo_suffix += f"_deltaMappingLin{str(DELTA_MAPPING_QTENC_SIM_THRESH).replace('.', '')}"
    if (load_new_delta_classes  or add_new_delta_classes) and 'tpch' not in Config.db_name:
        res_combo_suffix += f'_fq{FTUNE_QUERY_COUNT}'
        res_combo_suffix += f'_fe{FTUNE_EPOCHS}'
    
    res_combo_suffix += f'_dtbst{str(TB_SELECT_THRESHOLD).replace(".", "")}' if dynamic_tb_thresh else f'_tbst{str(TB_SELECT_THRESHOLD).replace(".", "")}'
    res_combo_suffix += '' if (tb_alpha == 0.05 and '--TBAlpha' not in received_args) else f'a{str(tb_alpha).replace(".", "")}'
    res_combo_suffix += '_skip' if SKIP_PREFETCH_STEP else ''
    res_combo_suffix += '_at' if IS_ANALYZING_TIME else ''

    query_template_encoding_method = 'plan'
    if '--qjson' in received_args:
        query_template_encoding_method = 'qjsn'

    model_suffix += f'_wqtenc_{query_template_encoding_method}' if use_plan else ''
    model_suffix += f'_all_hists' if use_all_hists else ''
    model_suffix += f'_biDelta' if use_binary_delta else ''
    model_suffix += f'_mixenc' if use_mixed_enc_for_q else ''
    model_suffix += f'_lstmenc' if use_lstm_encoding_for_qenc else ''
    model_suffix += f'_regulateTrain' if regulate_train_size else ''
    model_suffix += f'_twiceCls' if twice_class_size else ''
    model_suffix += f'_focal' if '--focal' in received_args else ''
    model_suffix += f'_tfocal' if '--tfocal' in received_args else ''
    model_suffix += f'_acb' if '--acb' in received_args else ''
    model_suffix += '_loadDelC' if load_new_delta_classes else ''
    model_suffix += '_addDelC' if add_new_delta_classes else ''
    model_suffix += 'p' if  (FQTESTSUFFIX != '') else ''
    model_suffix += f'_PPL{PLAN_PREP_LEVEL}' if PLAN_PREP_LEVEL != 1 else ''

    mbn = 'grasp_{model_suffix}{pred_model}_lowlr_b128'
    
    if Config.look_back != 2 or '--lookback' in received_args:
        mbn += f'_L{Config.look_back}'
    
    model_base_names = []
    if '--focalConfig' in received_args:
        if '--alpha' in received_args:
            alpha = float(received_args['--alpha'])
            gamma = int(received_args['--gamma'])

            ms = f'{model_suffix}_focala{str(alpha).replace(".", "")}g{str(gamma).replace(".", "")}'
            mbn = mbn.format(model_suffix=ms, pred_model=pred_model)
            model_base_names.append([mbn, alpha, gamma])
            
        else:
            for alpha in [.1, .25, .5, .75, 1]:
                for gamma in [1, 2, 3, 5]:
                    ms = f'{model_suffix}_focala{str(alpha).replace(".", "")}g{str(gamma).replace(".", "")}'
                    mbn = f'semantic_adr_septbdelta{ms}{pred_model}'
                    model_base_names.append([mbn, alpha, gamma])
                
    else:
        mbn = mbn.format(model_suffix=model_suffix, pred_model=pred_model)
        model_base_names.append([mbn, 1, 0])

    if '--dcCount' in received_args:
        dc_counts = [int(received_args['--dcCount'])]
    else:
        dc_counts = [1250]
    
    plan_types = [received_args.get('--planType', 'qjsn')]
    
    if '--k' in received_args:
        k_option = [int(received_args['--k'])]
    else:
        k_option = [100]
    
    if '--pSize' in received_args:
        received_args['--pSize'] = [int(received_args['--pSize'])]
        k_option[0] = int(k_option[0] * 32/received_args['--pSize'][0])

    if '--cSize' in received_args:
        received_args['--cSize'] = [float(received_args['--cSize']) if '0.' in received_args['--cSize'] else int(received_args['--cSize'])]
        cache_GBs = received_args['--cSize']

    tpch_initial_db= ''
    for plan_type in plan_types:
        query_template_encoding_method = plan_type
        for dc_count in dc_counts:
            for model_base_name, alpha, gamma in model_base_names:
                print(model_base_name, alpha, gamma)
                ALPHA = alpha
                GAMMA = gamma
                model_base_name = re.sub(r'(wqtenc_)([^_]+)', r'\1' + plan_type, model_base_name)
                print(model_base_name)
                if dc_count != 1250:
                    model_base_name = f'{model_base_name}_dc{dc_count}'
                if '--grasp' in received_args:
                    res_multiplier_opts = rs
                    par_sizes = received_args.get('--pSize', [32])

                    if IS_ANALYZING_TIME:
                        do_load_model = False
                        check_model_existence = False

                for initial_db in db_names:
                    if 'tpch' in initial_db:
                        tpch_initial_db = initial_db
                        if generalization_test:
                            initial_db = 'tpch_sf10_z05'
                        
                    if 'auction' in initial_db or initial_db in ['birds', 'genomic'] or 'wiki' in initial_db:
                        if '--cSize' not in received_args:
                            cache_GBs = [4]
                        if '--rs' not in received_args:
                            res_multiplier_opts = [40]

                    else:
                        if '--cSize' not in received_args:
                            cache_GBs = [8]
                        if '--rs' not in received_args:
                            res_multiplier_opts = [25]

                    for output_type in output_types:
                        for res_multiplier in res_multiplier_opts:
                            for par_size in par_sizes:
                                Config.max_partition_size = par_size

                                sec_tb_manager, sec_par_manager, sec_aff_mat_manager = None, None, None
                                init_tb_manager, init_par_manager, init_aff_mat_manager = None, None, None

                                try:

                                    if Config.db_name != initial_db:
                                        alter_config(initial_db, par_size)

                                    config_suffix = config_suffix_
                                    suffix = config_suffix
                                    psize = Config.max_partition_size
                                    pid_selection_mode = 'last'
                                    if just_plot and not (done_db == '' or done_db != initial_db):
                                        continue

                                    if init_tb_manager is None:
                                        init_tb_manager, init_par_manager, init_aff_mat_manager = get_managers(suffix)
                                    
                                    res_file_temp = f'{model_base_name}{res_combo_suffix}{block_level_name_suffix}{"_{m}xresLim".format(m=res_multiplier)}_{output_type}_{pid_selection_mode}{suffix}'

                                    model_manager = GrASP_comb(
                                        model_name=f'{Config.db_name}_{model_base_name}{block_level_name_suffix}_{output_type}_{pid_selection_mode}{suffix}P{psize}', 
                                        res_file_name=f'{res_file_temp}{cache_GB}GB', 
                                        config_suffix=config_suffix, table_manager=init_tb_manager, partition_manager=init_par_manager, 
                                        aff_mat=init_aff_mat_manager, output_type=output_type, cache_size=cache_size, query_template_encoding_method=query_template_encoding_method)
                                    print(model_manager.model_name)

                                    if check_model_existence and not IS_ANALYZING_TIME:
                                        do_load_model = model_manager.check_model_exist()
                                        if '--ftuneEpch' in received_args:
                                            prev_model_name = model_manager.model_name
                                            if 'tpch' not in Config.db_name:
                                                ftuned_model_name = prev_model_name.replace(config_suffix, f'{config_suffix}_ftune_fq{FTUNE_QUERY_COUNT}_fe{FTUNE_EPOCHS}')
                                            else:
                                                ftuned_model_name = prev_model_name.replace(config_suffix, f'{config_suffix}_ft')
                                            model_manager.model_name = ftuned_model_name
                                            do_load_model2 = model_manager.check_model_exist()
                                            
                                            model_manager.model_name = prev_model_name

                                    if just_plot:
                                        model_manager.plot_delta_analysis()
                                        print('plotting')
                                        done_db = initial_db 
                                        continue

                                    if (res_multiplier == res_multiplier_opts[0] and not do_load_model) or IS_ANALYZING_TIME:
                                        model_manager.prepare_train_data()
                                        t1 = time.time()
                                        model_manager.create_model()
                                        t2 = time.time()
                                        model_manager.compile_and_fit()
                                        t3 = time.time()
                                        if IS_ANALYZING_TIME:
                                            with open(f'{Config.result_base_path}/logFiles/{Config.db_name}_log_data.txt', 'a') as outfi:
                                                outfi.write(f'\nmodel_{model_manager.model_name}_resfile_{model_manager.res_file_name} model creation time: {t2 - t1}')
                                                outfi.write(f'\nmodel_{model_manager.model_name}_resfile_{model_manager.res_file_name} training time: {t3 - t2}')
                                                outfi.write(f'\nmodel_{model_manager.model_name}_resfile_{model_manager.res_file_name} total training time: {t3 - t1}')
                                    else:
                                        try:
                                            model_manager.load_model_from_file()
                                        except FileNotFoundError:
                                            err = traceback.format_exc()
                                            if 'qt_encoders' in err:
                                                print('Loading model failed due to qt_encoder issue. recreating the qt_encoder model.')
                                                model_manager.prepare_train_data(up_to='qt_enc')
                                                model_manager.load_model_from_file()
                                            else:
                                                raise FileNotFoundError    

                                    if generalization_test:
                                        org_suffix = model_manager.config_suffix
                                        alter_config(target_db, max_par_size=par_size)

                                        if 'ftune' in config_suffix_:
                                            sf = get_sf(initial_db)
                                            suffix = config_suffix_.replace('SFd', f'SF{sf}')
                                            config_suffix = suffix
                                            
                                        if 'NewMLPft' in suffix and initial_db != 'sdss_4':
                                            suffix = suffix.replace('NewMLPft', f'NewMLPft_ftune_frmSF{get_sf(initial_db)}')
                                            model_manager.config_suffix = suffix

                                        else:
                                            print(f'!! {suffix} need no change !!')

                                        if sec_tb_manager is None:
                                            sec_tb_manager, sec_par_manager, sec_aff_mat_manager = get_managers(suffix)
                                        model_manager.table_manager = sec_tb_manager
                                        model_manager.partition_manager = sec_par_manager
                                        if load_new_delta_classes or add_new_delta_classes:
                                            if Config.db_name == 'sdss_4':
                                                delta_db = 'sdss_4'
                                                model_name_for_new_classes = model_manager.model_name.replace(initial_db, 'sdss_4')#.replace('_MLPpred', '_twiceCls_MLPpred')
                                            elif Config.db_name == 'wiki_sf100_2':
                                                delta_db = 'wiki_sf100_1'
                                                model_name_for_new_classes = model_manager.model_name.replace(initial_db, 'wiki_sf100_1') 
                                            elif 'tpch' in Config.db_name:
                                                delta_db = tpch_initial_db
                                                model_name_for_new_classes = model_manager.model_name.replace(initial_db, tpch_initial_db)  #not sure if it is necessary
                                            else: 
                                                delta_db = Config.db_name.replace('1v2', '1v1')
                                                model_name_for_new_classes = model_manager.model_name.replace(initial_db, Config.db_name.replace('1v2', '1v1'))
                                            
                                            model_name_for_new_classes = get_ftune_model_name(model_manager.model_name, delta_db, dc_count, output_type)
                                            print(f'loading the classes from {model_name_for_new_classes}')
                                            model_manager.load_new_classes(model_name_for_new_classes)
                                            if 'tpch_sf30' in delta_db:
                                                model_manager.fine_tune(suffix, delta_db)
                                            else:
                                                model_manager.fine_tune(org_suffix, delta_db)

                                    for cache_GB in cache_GBs:
                                        cache_size = 16500 * cache_GB 
                                        print(f'Testing with {cache_GB}GB cache')
                                        if generalization_test:
                                            if 'auction' in Config.db_name:
                                                target_sf = 50
                                                if initial_db == 'auctionl1v1':
                                                    sf = target_sf
                                                else:
                                                    sf = get_sf(initial_db)                                        
                                            elif 'tpcc' in Config.db_name:
                                                target_sf = 250
                                                if initial_db == 'tpcc1v1':
                                                    sf = target_sf
                                                else:
                                                    sf = get_sf(initial_db)
                                            elif 'wiki' in Config.db_name:
                                                target_sf = 100
                                                sf = get_sf(initial_db)
                                            elif 'tpch' in Config.db_name:
                                                target_sf = 30
                                                sf = get_sf(initial_db)
                                            elif generalization_test and 'sdss' in Config.db_name:
                                                target_sf = 150
                                                sf = get_sf(initial_db)

                                            model_manager.set_db_scales(target_sf, sf)
                                            model_manager.res_file_name = f'{model_base_name}_TNsf{sf}TTsf{target_sf}_f{res_combo_suffix}{block_level_name_suffix}{"_{m}xresLim".format(m=res_multiplier)}_{output_type}_{pid_selection_mode}{suffix}{cache_GB}GB'
                                        
                                        else:
                                            model_manager.res_file_name = f'{res_file_temp}{cache_GB}GB'


                                        model_manager.cache_size = 16500 * cache_GB
                                        model_manager.test_model()

                                        curr_resf_name = model_manager.res_file_name


                                except Exception as e:
                                    traceback_info = traceback.format_exc()
                                    print(f"An error occurred while processing {config_suffix}:\n{traceback_info}")