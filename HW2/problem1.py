import pandas as pd

def process(filename):
    df = pd.read_csv("../Data/data/" + filename)
    df = df.drop(['index'], axis = 1)
    df[filename.split("_")[0] + '_bmi'] = df[filename.split("_")[0] + '_bmi'] / 10
    df[filename.split("_")[0] + '_stature_mm'] = df[filename.split("_")[0] + '_stature_mm'] / 1000
    return df

if __name__ == "__main__":
    maletrain = process("male_train_data.csv")
    femaletrain = process("female_train_data.csv")
    
    print(femaletrain['female_bmi'].head(10))
    print(femaletrain['female_stature_mm'].head(10))
    print(maletrain['male_bmi'].head(10))
    print(maletrain['male_stature_mm'].head(10))
