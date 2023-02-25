#!/usr/bin/env python
# coding: utf-8

# Housing price prediction
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to a north-south railroad. House price negotiations often have a lot of influencing factors and not just the number of bedrooms or the position of the kitchen.
# 
# Take the given dataset with 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. In this hackathon, predict the final price of each home. 
# 
# The application should be modeled using Machine Learning, you may explore libraries such as PySpark. Apply containerization principles as a better software engineering practice. You may explore Kafka server for streaming the data.
# 
# The model can be deployed using Docker containers for scalability.
# 
# Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/
# 
# Keywords: ML at scale, feature engineering, regression, random forest, gradient boosting, Distributed ML, Spark, Kafka, Containers

# from IPython.core.display import HTML
# display(HTML("<style>pre { white-space: pre !important; }</style>"))

# In[1]:


import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pyspark.sql.types as T
import pyspark.sql.functions as F
from functools import reduce
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.feature import StringIndexer,OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import RobustScaler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.regression import LinearRegression,LinearRegressionModel
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VarianceThresholdSelector
from pyspark.ml.feature import UnivariateFeatureSelector

### all the CONSTANTS used in the program

TRAIN_FILE = "house-prices/train.csv"
SDV_TRAIN_FILE = "house-prices/train_sdv.csv"
TEST_FILE = "house-prices/test.csv"
PREDICTION_OUTPUT_LOCATION_PROD = "/localuser/"
PREDICTION_OUTPUT_LOCATION_DEV = "/home/jovyan/work/"
TEST_STRUCT_FIELDS =  [
 StructField('Id', IntegerType(), True),
 StructField('MSSubClass', IntegerType(), True),
 StructField('MSZoning', StringType(), True),
 StructField('LotFrontage', IntegerType(), True),
 StructField('LotArea', IntegerType(), True),
 StructField('Street', StringType(), True),
 StructField('Alley', StringType(), True),
 StructField('LotShape', StringType(), True),
 StructField('LandContour', StringType(), True),
 StructField('Utilities', StringType(), True),
 StructField('LotConfig', StringType(), True),
 StructField('LandSlope', StringType(), True),
 StructField('Neighborhood', StringType(), True),
 StructField('Condition1', StringType(), True),
 StructField('Condition2', StringType(), True),
 StructField('BldgType', StringType(), True),
 StructField('HouseStyle', StringType(), True),
 StructField('OverallQual', IntegerType(), True),
 StructField('OverallCond', IntegerType(), True),
 StructField('YearBuilt', IntegerType(), True),
 StructField('YearRemodAdd', IntegerType(), True),
 StructField('RoofStyle', StringType(), True),
 StructField('RoofMatl', StringType(), True),
 StructField('Exterior1st', StringType(), True),
 StructField('Exterior2nd', StringType(), True),
 StructField('MasVnrType', StringType(), True),
 StructField('MasVnrArea', IntegerType(), True),
 StructField('ExterQual', StringType(), True),
 StructField('ExterCond', StringType(), True),
 StructField('Foundation', StringType(), True),
 StructField('BsmtQual', StringType(), True),
 StructField('BsmtCond', StringType(), True),
 StructField('BsmtExposure', StringType(), True),
 StructField('BsmtFinType1', StringType(), True),
 StructField('BsmtFinSF1', IntegerType(), True),
 StructField('BsmtFinType2', StringType(), True),
 StructField('BsmtFinSF2', IntegerType(), True),
 StructField('BsmtUnfSF', IntegerType(), True),
 StructField('TotalBsmtSF', IntegerType(), True),
 StructField('Heating', StringType(), True),
 StructField('HeatingQC', StringType(), True),
 StructField('CentralAir', StringType(), True),
 StructField('Electrical', StringType(), True),
 StructField('1stFlrSF', IntegerType(), True),
 StructField('2ndFlrSF', IntegerType(), True),
 StructField('LowQualFinSF', IntegerType(), True),
 StructField('GrLivArea', IntegerType(), True),
 StructField('BsmtFullBath', IntegerType(), True),
 StructField('BsmtHalfBath', IntegerType(), True),
 StructField('FullBath', IntegerType(), True),
 StructField('HalfBath', IntegerType(), True),
 StructField('BedroomAbvGr', IntegerType(), True),
 StructField('KitchenAbvGr', IntegerType(), True),
 StructField('KitchenQual', StringType(), True),
 StructField('TotRmsAbvGrd', IntegerType(), True),
 StructField('Functional', StringType(), True),
 StructField('Fireplaces', IntegerType(), True),
 StructField('FireplaceQu', StringType(), True),
 StructField('GarageType', StringType(), True),
 StructField('GarageYrBlt', IntegerType(), True),
 StructField('GarageFinish', StringType(), True),
 StructField('GarageCars', IntegerType(), True),
 StructField('GarageArea', IntegerType(), True),
 StructField('GarageQual', StringType(), True),
 StructField('GarageCond', StringType(), True),
 StructField('PavedDrive', StringType(), True),
 StructField('WoodDeckSF', IntegerType(), True),
 StructField('OpenPorchSF', IntegerType(), True),
 StructField('EnclosedPorch', IntegerType(), True),
 StructField('3SsnPorch', IntegerType(), True),
 StructField('ScreenPorch', IntegerType(), True),
 StructField('PoolArea', IntegerType(), True),
 StructField('PoolQC', StringType(), True),
 StructField('Fence', StringType(), True),
 StructField('MiscFeature', StringType(), True),
 StructField('MiscVal', IntegerType(), True),
 StructField('MoSold', IntegerType(), True),
 StructField('YrSold', IntegerType(), True),
 StructField('SaleType', StringType(), True),
 StructField('SaleCondition', StringType(), True),
]


COLS_TO_UPPER = ['ExterQual', 'ExterCond', 'KitchenQual', 'FireplaceQu', 'GarageFinish',
                 'GarageQual', 'GarageCond', 'PoolQC', 'BsmtQual', 'BsmtCond',
                 'HeatingQC', 'BsmtFinType1', 'BsmtFinType2','BsmtExposure','MSSubClass','SaleCondition','Fence']

COLS_TO_INT = ['BsmtFinType1', 'BsmtFinType2','ExterQual', 'ExterCond', 'KitchenQual', 'FireplaceQu','GarageQual', 'PoolQC',
               'BsmtQual','BsmtCond','HeatingQC','GarageCond','GarageFinish','BsmtExposure','MSSubClass','SaleCondition','Fence']

QUALITY_COLS =['ExterQual', 'ExterCond', 'KitchenQual', 'FireplaceQu','GarageQual', 'PoolQC','BsmtQual','BsmtCond','HeatingQC','GarageCond']

ORDINAL_CATEGORY_FEATURES = ['GarageYrBlt','MoSold','YrSold']

EXCLUDE_COLS  = ['SalePrice','LogSalePrice','Id']
BASEMENT_FINISH_CONVERTER ={ 'NO BASEMENT':'1', 'NA': '1', 'UNF':'2', 'LWQ':'3','REC':'4','BLQ':'5','ALQ':'6','GLQ':'7'}
GARAGE_FINISH_MAP={'NA': '1', "UNF": '2', "RFN": '3', "FIN": '4'} 
QUALITY_CONVERTER = {'NA':'1','NO FIREPLACE':'1','NO POOL':'1','NO BASEMENT':'1','NO GARAGE':'1',
                     'PO':'2', 'FA':'3','TA':'4' ,'GD':'5','EX' :'6'}

BSMTEXPOSURE_MAP = { 'NA':'1' ,'NO':'2','MN':'3','AV':'4','GD':'5'}
MSSUBCLASS_MAP = {'30':'1','180':'2','45':'2','190':'2','90':'2','150':'2','160':'2',
                  '50':'3','85':'3','40':'3','70':'4','80':'4','20':'4','75':'4','120':'4','60':'5'}
SALECONDITION_MAP ={'ABNORML': '2', 'ALLOCA': '2', 'ADJLAND': '2', 'FAMILY': '2', 'NORMAL': '1', 'PARTIAL': '1'}
FENCE_MAP ={'NA': '1', "MNWW": '2', "GDWO": '3', "MNPRV": '4', "GDPRV": '5'} 

TRAIN_SCHEMA = StructType(TEST_STRUCT_FIELDS + [StructField('SalePrice', IntegerType(), True)])
TEST_SCHEMA = StructType(TEST_STRUCT_FIELDS)

IMPUTE_COLS =[ 'LotFrontage', 'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                        'BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea']

def df_shape(spark_df):
    return spark_df.count(),len(spark_df.columns)

def get_dataframe(spark_session,filename,schema):
    """Load file as spark data frame """
    df = spark_session.read.csv(filename,schema = schema, header= True)
    return df

def combine_dfs(all_dfs):
    df= reduce(DataFrame.unionAll, all_dfs)
    return df

def get_null_counts(py_df):
    """Computes the value count of each column and reports null value count """
    results = {}
    for cname,ctype in py_df.dtypes:      
        results[cname] = py_df.where( py_df[cname].isNull() ).count()    
    null_counts= {k:v for k,v in results.items() if v!=0 }  
    return null_counts


def impute_nulls(df):
    """function to Impute nulls and returns dataframe"""
    df = df.na.fill(value=0,subset=IMPUTE_COLS)
    return df 

def add_new_features(df):
    """function to add new features and returns dataframe"""
    df = df.withColumn("TotalBath", df["FullBath"] + 0.5*df["HalfBath"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"])
    df = df.withColumn("TotalArea", df["GrLivArea"] + df["TotalBsmtSF"])
    df = df.withColumn("TotalFloorSF",df['1stFlrSF'] + df['2ndFlrSF'])
    df = df.withColumn("RemodelledAge",  df['YearRemodAdd']-df['YearBuilt'])
    df = df.withColumn("Age",  2010 -df['YearBuilt'])
    df = df.withColumn("IsRegularLotShape" , F.when(df.LotShape ==  "Reg","1").otherwise("0")) 
    df = df.withColumn("IsRemodeled" , F.when(df.YearBuilt !=  df.YearRemodAdd,"1").otherwise("0")) 
    df = df.withColumn("VeryNewHouse" , F.when(df.YearBuilt ==  df.YrSold,"1").otherwise("0")) 
    df = df.withColumn("TotalPorchSF", df["OpenPorchSF"] + df["EnclosedPorch"]+df["3SsnPorch"] + df["ScreenPorch"])
    df = df.drop("FullBath","HalfBath","BsmtFullBath","BsmtHalfBath","1stFlrSF","2ndFlrSF","GrLivArea","TotalBsmtSF")
    df = df.drop('OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','YearRemodAdd','YearBuilt')
    return df

def convert_features(df):
    """function to convert features and returns dataframe"""
    for col_name in COLS_TO_UPPER:
        df =  df.withColumn(col_name,F.upper(F.col(col_name)))
        
    df = df.replace(BASEMENT_FINISH_CONVERTER,subset=['BsmtFinType1', 'BsmtFinType2'])
    df = df.replace(QUALITY_CONVERTER,subset=QUALITY_COLS)
    df = df.replace(GARAGE_FINISH_MAP,subset=['GarageFinish'])
    df = df.replace(BSMTEXPOSURE_MAP,subset=['BsmtExposure'])
    df = df.replace(MSSUBCLASS_MAP,subset=['MSSubClass'])
    df = df.replace(SALECONDITION_MAP,subset=['SaleCondition'])
    df = df.replace(FENCE_MAP,subset=['Fence'])
    
    for col in COLS_TO_INT:
        df = df.withColumn(col,F.col(col).cast('integer'))
    return df


def add_log_sale_price(df):
    """function to add column LogSalePrice and returns dataframe"""
    df = df.withColumn("LogSalePrice", F.log10(F.col("SalePrice")))
    return df

    
def get_cat_and_cont_features(df,exclude_cols=None):
    """function to return categorical and continous features"""
    category_features = [f.name for f in df.schema.fields if isinstance(f.dataType, T.StringType)]
    ignore_cols  = category_features + exclude_cols if exclude_cols else category_features
    category_features = category_features + ORDINAL_CATEGORY_FEATURES
    continuous_features = [ f.name for f in df.schema.fields if f.name not in ignore_cols]
    return category_features,continuous_features
    

def create_pipeline(category_features,continuous_features,add_vector_index=False):
    """function to create pipeline based on categorical and continous features and returns pipeline"""
    str_enc = [ f"{col}_str_enc" for col in category_features]
    ohe_enc = [ f"{col}_ohe" for col in category_features]
    stage_str_enc = [StringIndexer(inputCol=a, outputCol= b,handleInvalid="keep") for a,b in zip(category_features,str_enc)]
    stage_ohe_enc = [OneHotEncoder(inputCol= a, outputCol= b) for a,b in zip(str_enc,ohe_enc) ]
    next_input_col = 'indexed_features' if add_vector_index else 'features'
    
    if add_vector_index:
        stage_vec_assembler = VectorAssembler(inputCols= continuous_features + str_enc ,outputCol="features")
        stage_indexer =    VectorIndexer(inputCol= "features", outputCol= "indexed_features",handleInvalid="keep")
    else:
        stage_vec_assembler = VectorAssembler(inputCols= continuous_features + ohe_enc ,outputCol="features")
    
        
    stage_scaler = RobustScaler(inputCol=next_input_col,outputCol="scaled_features")
    stage_selector = UnivariateFeatureSelector(featuresCol='scaled_features',
                                           outputCol="selected_features",
                                           labelCol="SalePrice",
                                           )

    stage_selector.setFeatureType("continuous").setLabelType("continuous") 
    if add_vector_index:
        pipeline_stages  = stage_str_enc + [stage_vec_assembler,stage_indexer,stage_scaler]
    else:
        pipeline_stages  = stage_str_enc + stage_ohe_enc + [stage_vec_assembler,stage_scaler]
        
    pipeline = Pipeline(stages= pipeline_stages)
    return pipeline

def test_model(env="PROD"):
    """function to test the model save the results to csv """
    OUTPUT_LOCATION = PREDICTION_OUTPUT_LOCATION_DEV if env =="DEV" else PREDICTION_OUTPUT_LOCATION_PROD
    PREDICTION_FILE = f"{OUTPUT_LOCATION}"+"predictions.csv"
    print("predictions will be stored at:",PREDICTION_FILE)
    spark_session = SparkSession.builder.appName("HousePrices").getOrCreate()
    print('Starting model testing')
    df = get_dataframe(spark_session,TEST_FILE,TEST_SCHEMA)
    print('loaded data into spark df')
    df = impute_nulls(df)
    print('finished imputing nulls ')
    df = add_new_features(df)
    print('finished adding new features')
    df = convert_features(df)
    print('finished adding converting features')
 
    category_features,continuous_features = get_cat_and_cont_features(df,EXCLUDE_COLS)
    print('finished getting category and continous features')
    pipeline_model = PipelineModel.load('pipeline_model.h5')
    print('finished loading pipeline')
    lr_model = LinearRegressionModel.load('lr_model.h5')
    test_pdf = pipeline_model.transform(df)
    test_prediction= lr_model.transform(test_pdf)
    test_prediction = test_prediction.select("Id","prediction")
    test_prediction = test_prediction.withColumnRenamed("prediction","SalePrice")
    print(test_prediction.show(5))
    #test_prediction.repartition(1).write.format('csv').mode("overwrite").options(sep=',', header='true').save(PREDICTION_FILE)
    pred_df = test_prediction.toPandas()
    pred_df.to_csv(PREDICTION_FILE,index=False)
    print('Finished model testing')
    
#master("local[*]").config('job.local.dir', 'file:/home/joyvan/work')
def train_model(synthetic_data=False):
    """function to test the model save the results to csv """
    spark_session = SparkSession.builder.appName("HousePrices").getOrCreate()
    print('Starting model training')
    df = get_dataframe(spark_session,TRAIN_FILE,TRAIN_SCHEMA)
    print('loaded training data into spark df')
    
    if synthetic_data:
        print("Shape of training data before adding synthentic data:", df_shape(df))
        sdf = get_dataframe(spark_session,SDV_TRAIN_FILE,TRAIN_SCHEMA)
        print('loaded synthetic training data into spark df')
        df = combine_dfs([df,sdf])
        print('combined data into spark df')
        print("Shape of training data after adding synthentic data:", df_shape(df))
        
    df = impute_nulls(df)
    print('finished imputing nulls ')
    df = add_new_features(df)
    print('finished adding new features')
    df = convert_features(df)
    print('finished adding converting features')
    category_features,continuous_features = get_cat_and_cont_features(df,EXCLUDE_COLS)
    print('finished getting category and continous features')
    null_counts = get_null_counts(df)
    if null_counts:
        print(null_counts)
        raise ValueError("Dataset has nulls, cannot create model")
   

    pipeline  = create_pipeline(category_features,continuous_features)
    print('finished creating pipeline')
    
   
    train_df, test_df =  df.randomSplit([0.8,0.2], seed = 42)
    pipeline_model = pipeline.fit(train_df)
    train_pdf = pipeline_model.transform(train_df)
    print('finished pipeline transforming training data')
    lr =  LinearRegression(featuresCol="scaled_features", labelCol="SalePrice",
                           maxIter= 10,regParam=0.3, elasticNetParam=0.8)
    
    print(f"Observations in training set = {train_df.count()}")
    print(f"Observations in testing set = { test_df.count()}")
    lr_model = lr.fit(train_pdf)
    print('finished training model')
    trainingSummary = lr_model.summary
    print(trainingSummary.residuals.show()) 
    print(f"RMSE: {trainingSummary.rootMeanSquaredError:.4f}")
    print(f"r2: {trainingSummary.r2:.4f}")
    test_pdf = pipeline_model.transform(test_df)
    test_prediction= lr_model.transform(test_pdf)
    eval_lr = RegressionEvaluator(predictionCol='prediction', labelCol='SalePrice')
    rmse= eval_lr.evaluate(test_prediction, {eval_lr.metricName:'rmse'})
    r2 =eval_lr.evaluate(test_prediction,{eval_lr.metricName:'r2'})
    print(f"RMSE: {rmse:.4f}")
    print(f"r2: {r2:.4f}")
    print('Finished model creation')
    return pipeline_model,lr_model


if __name__ =="__main__":
    test_model()
    
