#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
#from statistics import mode, median

#from pprint import pprint
#import matplotlib.pyplot as plt
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import mean_squared_error, mean_absolute_error


# from google.colab import drive
# drive.mount("/content/drive")

# import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')

class DecisionTree:
    tree = {}
    
    def find_for_numerical_data(self, row, col_name, val):
        if row[col_name] <= float(val):
            return 0 #left branch
        else:
            return 1


    def find_for_categorical_data(self, row, col_name, val):
        if row[col_name] == val:
            return 0    
        else:
            return 1 


    def map_col_to_index(self, df):
        global col_to_idx        
        i = 0
        for col_nm in df.columns:
            col_to_idx[i] = col_nm
            i = i + 1


    def is_numerical(self, feature_nm):
        numerical_features =['YrSold','MoSold','MiscVal','PoolArea','ScreenPorch','3SsnPorch','EnclosedPorch','OpenPorchSF',
                             'WoodDeckSF','GarageArea','GarageCars','GarageYrBlt','Fireplaces','TotRmsAbvGrd','Kitchen','Bedroom',
                             'LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                             'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath',
                             'HalfBath']
        
        return feature_nm in numerical_features


    def util_mse(self, x):   
        if(x.shape[0] == 0):
            return 0;     
        mean_x = x.mean()
        return np.square([x1 - mean_x for x1 in x]).sum()


    def find_mean_squared_error(self, x, y): #x and y are numpy 2d arrays
        x = x[:, -1]
        y = y[:, -1]

        mean_of_left_data = self.util_mse(x)
        mean_of_right_data = self.util_mse(y)

        total_len = len(x) + len(y)
        weighted_mean = (len(x) / total_len * mean_of_left_data) + (len(y) / total_len * mean_of_right_data)

        # print("weighted_mean: ",weighted_mean)
        return weighted_mean


    def drop_columns(self, df):
        # print(df.info())
        # observed the columns which had NAN in >1/2 the no. of rows of train file and dropped them
        df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
        # print(df.info())
        return df


    def clean_data(self, df):
        df = self.drop_columns(df)
        for col_name in df.columns:
            if(self.is_numerical(col_name)):
                df[col_name].fillna(df[col_name].mean(), inplace=True)
            else:
                df[col_name].fillna(df[col_name].mode()[0], inplace=True)
                # mode()[0]: reqd when there are mulitple modes (val occuring equal no. of times..)
        return df


    def train_validation_split(self, df):
        dataLen=int(1*df.shape[0])
        return df.iloc[0:dataLen], df.iloc[dataLen:,0:]


    def is_pure(self, data):#data: a numpy 2D array
        sales_data_label=data[:,-1]
        set_label=set(sales_data_label);
        if(len(set_label)>1):
            return False
        else:
            return True


    def find_potential_splits(self, data):
        global col_to_idx
        potential_split_dict = {}
        for col_idx in range(0, data.shape[1]-1):
            unique_col=np.unique(data[:, col_idx])
            potential_split_dict[col_idx]=[]
            if self.is_numerical(col_to_idx[col_idx]):
                for row_idx in range(1, unique_col.shape[0]):
                    prev = unique_col[row_idx-1]
                    nxt = unique_col[row_idx]
                    potential_split_dict[col_idx].append((prev+nxt)/2)
            else:
                potential_split_dict[col_idx].extend(unique_col) #don't use append; use extend

        return potential_split_dict


    def split_data(self, split_col_idx, split_val, data):
        global col_to_idx
        col_name=col_to_idx[split_col_idx]
        col=data[:, split_col_idx]

        if(self.is_numerical(col_name)):
            data_left = data[col <= split_val]
            data_right = data[col > split_val]
        else:
            data_left = data[col == split_val]
            data_right = data[col != split_val]

        return data_left, data_right

    
    def find_best_split(self, potential_split_dict, data):
        split_col = None
        split_val = None
        overall_mse = None
        for col_idx in potential_split_dict:
            for split_value in potential_split_dict[col_idx]:
                data_left, data_right = self.split_data(col_idx, split_value, data)                
                mse = self.find_mean_squared_error(data_left, data_right)
                if(overall_mse is None):                    
                    overall_mse = mse
                    split_col = col_idx
                    split_val = split_value
                else:
                    if mse < overall_mse:
                        overall_mse = mse
                        split_col = col_idx
                        split_val = split_value
        return split_col, split_val


    def build_decision_tree(self, data, depth, max_depth):
        if((depth == max_depth) or self.is_pure(data)):
            return data[:,-1].mean()
        else:
            potential_split_dict = self.find_potential_splits(data)
            best_split_col, best_split_val = self.find_best_split(potential_split_dict, data)
            # print("best_split_col, best_split_val ", best_split_col, best_split_val)
            data_left, data_right = self.split_data(best_split_col, best_split_val, data)

            left_branch = self.build_decision_tree(data_left, depth + 1, max_depth)
            right_branch = self.build_decision_tree(data_right, depth + 1, max_depth)

            global col_to_idx
            col_name = col_to_idx[best_split_col]
            tag=str(col_name)+" "+str(best_split_val)
            subtree = {}
            subtree[tag]=[]

            if(left_branch == right_branch):
                subtree = left_branch
            elif(left_branch != right_branch):
                subtree[tag].append(left_branch)
                subtree[tag].append(right_branch)
            return subtree


    def predict_from_tree(self, row, tree):
        root = list(tree.keys())[0]
        split_col_name, split_value = root.split(" ")

        if self.is_numerical(split_col_name):
            p = self.find_for_numerical_data(row, split_col_name, split_value)
        else:
            p = self.find_for_categorical_data(row, split_col_name, split_value)
        ans = tree[root][p]

        #base case
        if not isinstance(ans, dict):
            return ans        
        else:
            return self.predict_from_tree(row, ans)


    def predict_util(self, test_df):
        predicted = []
        for idx in range(test_df.shape[0]):
            # validation_label = test_df.iloc[idx, -1]
            # print("validation_label: ", validation_label)
            predicted_label = self.predict_from_tree(test_df.iloc[idx], self.tree)
            # print("predicted_label: ", predicted_label)
            predicted.append(predicted_label)
        return predicted


    def predict(self, test_file):
        test_df = pd.read_csv(test_file)[:]
        test_df=test_df.drop("Id", axis=1) #dropping the id col
        test_df = self.clean_data(test_df)
        return self.predict_util(test_df)


    def train(self, train_file_name):
        df=pd.read_csv(train_file_name)[:]
        df=df.drop("Id", axis=1) #dropping the id col
        df = self.clean_data(df)

        global col_to_idx
        col_to_idx = {}
        self.map_col_to_index(df)

        train_df, validation_df = self.train_validation_split(df)
        self.tree = self.build_decision_tree(train_df.values, depth=0, max_depth=5)
        # pprint(self.tree)

#dtree_regressor=DecisionTree()
#dtree_regressor.train('./Datasets/q3/train.csv')
#
#predictions = dtree_regressor.predict('./Datasets/q3/test.csv')
#test_labels = list()
#with open("./Datasets/q3/test_labels.csv") as f:
#  for line in f:
#    test_labels.append(float(line.split(',')[1]))
#print ("MSE:", mean_squared_error(test_labels, predictions))
#print("MAE:", mean_absolute_error(test_labels,predictions))

