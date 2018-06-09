import pandas as pd
import os

def main():
    for file in os.listdir(os.path.join("data","old")):
        df = pd.read_csv(os.path.join("data","old",file),sep=",")
        data_df = df[df.columns[:-2]]
        em_df = df[df.columns[-2]]
        file_name,ext = os.path.splitext(file)
        data_df.to_csv(os.path.join("data","new",file_name+".dat"),index=False)
        em_df.to_csv(os.path.join("data","new",file_name+".emo"),index=False)

if __name__ == '__main__':
    main()