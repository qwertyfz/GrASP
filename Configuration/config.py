import os


def read_table_list_from_file(file_path):
    with open(file_path) as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_table_lookups(table_lookup_file_path):
    lines = read_table_list_from_file(table_lookup_file_path)
    lookup = {}
    for i in range(1, len(lines)+1):
        lookup[lines[i-1]] = i
        lookup[i] = lines[i-1]
    return lines, lookup


class Config:
    is_on_server = True
    """ Database Config"""
    db_name = 'wiki_sf100_2'

    db_user = 'postgres'
    db_password = 'pass'

    db_host = '127.0.0.1'
    db_port = '5432'

    """ model config """
    encoding_length = 32
    encoding_epoch_no = 100
    look_back = 2
    do_string_encoding = True
    str_encoding_dim = 8
    is_test = False
    tb_encoding_method = ''
    config_suffix = 'NewMLP_sequential'

    """ system config """
    block_level_query_encoding = False
    prefetching_k = 40
    prefetching_augment_k = 0
    max_partition_size = 32
    logical_block_size = 8
    trace_win_size = 256 #10000 or 256
    read_ahead_trigger_threshold = 13
    extend_size = max_partition_size
    extend_type = 'table'
    adr_digit_num = 5

    allow_default_par = False
    skip_random_str_cols = False

    lock_file = './Locks/db_lock.lock'
    lock_request_file = './Locks/db_lock_request.lock'
    lock_requested = False
    lock_acquired = False
    use_locking = False

    csv_file_separator = '\|\|'
    if 'wiki' in db_name:
        csv_file_separator = '\t'
    elif db_name not in ['birds', 'genomic'] and 'sdss' not in db_name:
        csv_file_separator = '$'


    """ Config files """    
    base_dir = "./"
    result_base_path = './Results'

    pca_exclude_file_path = os.path.join(base_dir, 'Data/pcaExclude.txt')
    
    try:
        table_lookup_file_path = os.path.join(base_dir, f"Data/{db_name}_tableLookUp.txt")
        table_list, table_lookup = read_table_lookups(table_lookup_file_path)
    except FileNotFoundError:
        if 'auction' in db_name:
            general_db_name = 'auction'
        elif 'tpcc' in db_name:
            general_db_name = 'tpcc'
        elif 'benchbase' in db_name:
            general_db_name = 'benchbase'
        elif 'wiki' in db_name:
            general_db_name = 'wiki'
        elif 'tpch' in db_name:
            general_db_name = 'tpch'
        table_lookup_file_path = os.path.join(base_dir, f"Data/{general_db_name}_tableLookUp.txt")
        table_list, table_lookup = read_table_lookups(table_lookup_file_path)

    tb_bid_range = {}
    actual_tb_bid_range = {}
    tb_lba_offset = {}
    pca_exclude_tables = read_table_list_from_file(pca_exclude_file_path)

    def __init__(self):
        pass

def alter_config(dbname='sdss_1', max_par_size=64, tb_lookup_fp='/navi_tableLookUp.txt'):
    Config.db_name = dbname
    Config.max_partition_size = max_par_size
    Config.csv_file_separator = '\|\|'
    if 'wiki' in dbname:
        Config.csv_file_separator = '\t'
    elif dbname not in ['birds', 'genomic'] and 'sdss' not in dbname:
        Config.csv_file_separator = '$'
    try:
        Config.table_lookup_file_path = os.path.join(Config.base_dir, f"Data/{dbname}_tableLookUp.txt")
        table_list, table_lookup = read_table_lookups(Config.table_lookup_file_path)
    except FileNotFoundError:
        if 'auction' in dbname:
            general_db_name = 'auction'
        elif 'tpcc' in dbname:
            general_db_name = 'tpcc1v1'
        elif 'benchbase' in dbname:
            general_db_name = 'benchbase'
        elif 'wiki' in dbname:
            general_db_name = 'wiki'
        elif 'tpch' in dbname:
            general_db_name = 'tpch'
        print(dbname)
        Config.table_lookup_file_path = os.path.join(Config.base_dir, f"Data/{general_db_name}_tableLookUp.txt")
        table_list, table_lookup = read_table_lookups(Config.table_lookup_file_path)
    Config.table_list = table_list
    Config.table_lookup = table_lookup
