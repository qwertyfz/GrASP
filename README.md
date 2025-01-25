# GrASP

## GrASP: A Generalizable Address-based Semantic Prefetcher for Scalable Transactional and Analytical WorkloadsLearning Based Semantic Prefetching for Exploratory Database Workloads

Combining the idea of logical block addresses (LBA) delta modelling and semantic prefetching, we propose, GrASP, an address-based semantic prefetcher. 



##
## Steps to test GrASP
1. **Install the requirements** - All necessary packages and libraries are listed in the requirements.txt file. You can install them using the following command:
    ```sh
    pip install -r requirements.txt
    ```
2. **Setup the Database** - Create your database in PostgreSQL.
3. **Set up the config file** (Configuration/Config.py) - Provide the database connection details and set the configuration parameters values. The configuration parameters are set to default values and can be left unchanged.
4. **Prepare workload files** - The workload file should contain the following columns:

    | theTime | ClientIP | row | statement | resultBlock |
    | ------ | ------ | ------ | ------ | ------ | 
    03/23/2014 05:51:59 PM|140.1.2.0|10|select * from tbplasmiddna   where ngulwater < 0 |[tbplasmiddna_20, tbplasmiddna_18, tbplasmiddna_17, tbplasmiddna_19]
    
    The _resultBlock_ column, stores blocks accessed by the query statement. You can use `bid_getter.py` to get _resultBlock_ of a workload. This script executes the query after clearing the PostgreSQL and system cache and then checks the cache contents. Superuser access privileges are required to clear the caches.    
6. **Train GrASP** - With the following command, run GrASP with your desired initial configuration or arguments. When you run this code on a dataset for the first time, it will generate block encodings and save them for future use. Once the block encodings are collected, the code will generate model input and output and proceed to train the model. Finally, it will store the trained model for future testing.
    ```sh
    python3 -m Backend.Models.GrASP
    ```
A more detailed execution command example is:

    ```sh
    python3 -m Backend.Models.GrASP --Database="wiki_sf100_1" --predModel="_multiLSTM" --GPU="1" --focalConfig --alpha=0.75 --gamma=3 --dcCount=1500 --grasp --binary_delta --planType="qjsn" --qjson --k=100 --dynTBThresh --skip --epochs=25 --ftuneQC=5000 --ftuneEpch=25 --addDeltaCls
    ```
    
7. **Test GrASP** - Configure the test settings, such as cache size and prefetch size. This will load the model trained in the previous step and utilize it to make predictions on a provided workload.
8. **Test results** - All test results can be found in the Results folder and can be accessed to generate plots.
9. (optional) **Baseline prefetchers** - The baselilne prefetchers described in the paper have been implemented and can be tested by running the `main.py` and `selep_main.py` files.