import pandas as pd
import os
from env import get_db_url

"""
USAGE: 
Use `from wrangle import wrangle_zillow` at the top of your notebook.
This 

"""
def get_zillow_data():
    """Seeks to read the cached zillow.csv first """
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return get_new_zillow_data()

def get_new_zillow_data():
    """Returns a dataframe of all 2017 properties that are Single Family Residential"""

    sql = """
    select 
    bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    from properties_2017
    join propertylandusetype using (propertylandusetypeid)
    where propertylandusedesc = "Single Family Residential"
    """
    return pd.read_sql(sql, get_db_url("zillow"))


def handle_nulls(df):    
    # We keep 99.41% of the data after dropping nulls
    # round(df.dropna().shape[0] / df.shape[0], 4) returned .9941
    df = df.dropna()
    return df


def optimize_types(df):
    # Convert some columns to integers
    # fips, yearbuilt, and bedrooms can be integers
    df["fips"] = df["fips"].astype(int)
    df["year_built"] = df["year_built"].astype(int)
    df["bedrooms"] = df["bedrooms"].astype(int)    
    df["tax_value"] = df["tax_value"].astype(int)
    df["area"] = df["area"].astype(int)
    return df


def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
    df = df[df.bathrooms <= 6]
    
    df = df[df.bedrooms <= 6]

    df = df[df.tax_value < 2_000_000]

    return df

def rename_columns(df):
    # renaming columns
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 'calculatedfinishedsquarefeet':'area', 'taxvaluedollarcnt':'tax_value', 'taxamount':'tax_amount','yearbuilt':'year_built'})

    return df


def wrangle_zillow():
    """
    Acquires Zillow data
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    returns a clean dataframe
    """
    df = get_zillow_data()

    df = handle_nulls(df)

    df = optimize_types(df)

    df = handle_outliers(df)

    df = rename_columns(df)

    df.to_csv("zillow.csv", index=False)

    return df

def train_validate_test(df):
    '''
    Takes a dataframe and splits into train, validate, test 
    into 70%, 20%, 10% respectively.
    '''

    # Import to use split function, can only split two at a time
    from sklearn.model_selection import train_test_split

    # Frist, split into train + validate together and test by itself
    # Test will be about %10 of the data, train + validate is %70 for now
    # Set random_state so we can reproduce the same 'random' data
    train_validate, test = train_test_split(df, test_size = .10, random_state = 123)

    # Second, we plit train + validate into their seperate variables
    # Train will be about %70 of the data, Validate will be about %20 of the data
    train, validate = train_test_split(train_validate, test_size = .20, random_state = 123)

    # These two print functions allow us to ensure the date is properly split
    # Will print the shape of each variable when running the function
    print("train shape: ", train.shape, ", validate shape: ", validate.shape, ", test shape: ", test.shape)

    # Will print the shape of eachvariable as a percentage of the total data set
    # Varialbe to hold the sum of all rows (total observations in the data)
    total = df.count()[0]
    print("\ntrain percent: ", round(((train.shape[0])/total),2) * 100, 
            ", validate percent: ", round(((validate.shape[0])/total),2) * 100, 
            ", test percent: ", round(((test.shape[0])/total),2) * 100)

    return train, validate, test


## TODO Encode categorical variables (and FIPS is a category so Fips to string to one-hot-encoding
## TODO Scale numeric columns
## TODO Add train/validate/test split in here
## TODO How to handle 0 bedroom, 0 bathroom homes? Drop them? How many? They're probably clerical nulls