import numpy as np
import pandas as pd
import os, itertools, random, argparse
import warnings
warnings.filterwarnings(action='ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='test', choices=['train', 'test'], help='dataset to choose')

    args = parser.parse_args()
    print('--- Parameters ---')
    print(args)
    print('-' * 30)

    data = pd.read_csv(f'../dataset/{args.dataset}_motion_data.csv')

    X_data = data[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Timestamp']]
    data.Class = data.Class.replace(['AGGRESSIVE', 'NORMAL', 'SLOW'], [0, 1, 2])
    y_data = data[['Class']]

    X_data['Timestamp'] = np.arange(len(X_data))

    # Acceleration Magnitude
    X_data['AccMagnitude'] = np.sqrt(X_data['AccX']**2 + X_data['AccY']**2 + X_data['AccZ']**2)

    # Rotation Magnitude
    X_data['GyroMagnitude'] = np.sqrt(X_data['GyroX']**2 + X_data['GyroY']**2 + X_data['GyroZ']**2)

    # Jerk
    X_data['JerkX'] = X_data['AccX'].diff().div(X_data['Timestamp'].diff(), fill_value=0)
    X_data['JerkY'] = X_data['AccY'].diff().div(X_data['Timestamp'].diff(), fill_value=0)
    X_data['JerkZ'] = X_data['AccZ'].diff().div(X_data['Timestamp'].diff(), fill_value=0)
    X_data['JerkMagnitude'] = np.sqrt(X_data['JerkX']**2 + X_data['JerkY']**2 + X_data['JerkZ']**2)

    # Remove first data due to NaN value at Jerk variables
    # X_data_dropna = X_data.dropna()
    X_data = X_data.iloc[1:]
    # X_data = X_data_dropna
    y_data = y_data[1:]

    # Selecting Normal and Aggressive Data
    data_sum = pd.concat([X_data, y_data], axis=1)
    data_sum_agg = data_sum[data_sum['Class'] == 0]
    data_sum_norm = data_sum[data_sum['Class'] == 1]
    data_sum_agg_norm = pd.concat([data_sum_agg, data_sum_norm])

    # Final Split
    X_data_agg_norm = data_sum_agg_norm[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AccMagnitude', 'GyroMagnitude', 'JerkX', 'JerkY', 'JerkZ', 'JerkMagnitude']]
    y_data_agg_norm = data_sum_agg_norm[['Class']]
    
    # Saving dataset
    X_data_agg_norm.to_csv(f'../dataset/X_{args.dataset}.csv', index=False)
    y_data_agg_norm.to_csv(f'../dataset/y_{args.dataset}.csv', index=False)
    
    # Print Shape
    print('X_data Shape :', X_data_agg_norm.shape)
    print('y_data Shape :', y_data_agg_norm.shape)

    return X_data_agg_norm, y_data_agg_norm

if __name__ == '__main__':
    main()
