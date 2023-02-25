import pandas as pd
from sdv.tabular import GaussianCopula
from sdv.evaluation import evaluate


TRAIN_FILE = "house-prices/train.csv"
SDV_TRAIN_FILE = "house-prices/train_sdv.csv"
RESEQUNCE_START_ID = 8000
DEFAULT_ROWS =1500


def load_train_data(train_file):
    """function to load training file and return pandas dataframe """
    ### For reference , spark treats strings with NA as values rather than pandas in which treats them null 
    df = pd.read_csv(train_file,keep_default_na=False)
    return df


def generate_synthetic_data(df,num_rows=DEFAULT_ROWS):
    """function to generate synthetic data  and returns it as pandas dataframe """
    model = GaussianCopula(primary_key='Id')
    model.fit(df)
    syn_df = model.sample(num_rows=num_rows)
    return syn_df


def process_synthetic_data(df):
    """function to process synthentic data and returns pandas dataframe"""
    
    df['Id'] = df['Id'] + RESEQUNCE_START_ID
    return df

def evaluate_synthetic_data(df, syn_df):
    """function to process synthentic data and returns pandas dataframe"""
    eval_result  = evaluate(syn_df.drop('Id',axis=1), df.drop('Id',axis=1)) *100
    print(f"Generated synthenthic data matches with {eval_result:.2f} of the orginal dataset")

          
def run():
    """ main function to generate and save synthetic data"""      
    df = load_train_data(TRAIN_FILE)
    syn_df = generate_synthetic_data(df)   
    syn_df = process_synthetic_data(syn_df)
    evaluate_synthetic_data(df, syn_df)
    syn_df.to_csv(SDV_TRAIN_FILE,index=False)
    print("Sythentic data save to :",SDV_TRAIN_FILE)
          
if __name__ == '__main__':
    run()
          
          
 
          

     

              
    
    
    

