import pandas as pd
from sklearn.model_selection import train_test_split
from price_model import PriceModel
from price_model_log import PriceModelLog
from price_model_boost import PriceModelBoost
import sys


def read_data(filename):
	return pd.read_csv(filename, parse_dates=['host_since', 'first_review', 'last_review'])





if __name__ == "__main__":

	file_train = 'train.csv'
	data = read_data(file_train)

	X_df = data.drop('price', axis=1)
	y_df = data['price']


	mdl = PriceModel()
	mdl.train(X_df, y_df)



