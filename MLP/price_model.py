import pandas as pd
from sklearn.model_selection import train_test_split
import time
import numpy as np
import math
from mlp_model import MLP
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import proximal_gradient.proximalGradient as pg
import torch.optim.lr_scheduler as lr_scheduler
import random
from sklearn.metrics import root_mean_squared_error
device = torch.device('cuda')

class PriceModel:
	def __init__(self):

		# noramlization parameters
		self.property_type_set = set()
		self.neighbourhood_cleansed_set = set()
		self.neighbourhood_group_cleansed_set = set()
		self.latitude_extreme = [float('inf'), float('-inf')]
		self.longitude_extreme = [float('inf'), float('-inf')]
		self.host_since_extreme = [float('inf'), float('-inf')]
		self.host_verifications_set = set()
		self.host_response_time_set = set()
		self.host_response_rate_extreme = [float('inf'), float('-inf')]
		self.host_acceptance_rate_extreme = [float('inf'), float('-inf')]
		self.host_listings_count_extreme = [float('inf'), float('-inf')]
		self.host_total_listings_count_extreme = [float('inf'), float('-inf')]
		self.calculated_host_listings_count_extreme = [float('inf'), float('-inf')]
		self.calculated_host_listings_count_entire_homes_extreme = [float('inf'), float('-inf')]
		self.calculated_host_listings_count_private_rooms_extreme = [float('inf'), float('-inf')]
		self.calculated_host_listings_count_shared_rooms_extreme = [float('inf'), float('-inf')]
		self.room_type_set = set()
		self.accommodates_extreme = [float('inf'), float('-inf')]
		self.bathrooms_extreme = [float('inf'), float('-inf')]
		self.bedrooms_extreme = [float('inf'), float('-inf')]
		self.beds_extreme = [float('inf'), float('-inf')]
		self.amenities_set = set()
		self.avail_30_extreme = [float('inf'), float('-inf')]
		self.avail_60_extreme = [float('inf'), float('-inf')]
		self.avail_90_extreme = [float('inf'), float('-inf')]
		self.avail_365_extreme = [float('inf'), float('-inf')]
		self.minimum_nights_extreme = [float('inf'), float('-inf')]
		self.maximum_nights_extreme = [float('inf'), float('-inf')]
		self.num_reviews_extreme = [float('inf'), float('-inf')]
		self.num_reviews_ltm_extreme = [float('inf'), float('-inf')]
		self.num_reviews_l30d_extreme = [float('inf'), float('-inf')]
		self.first_review_extreme = [float('inf'), float('-inf')]
		self.last_review_extreme = [float('inf'), float('-inf')]
		self.review_scores_rating_extreme = [float('inf'), float('-inf')]
		self.review_scores_accuracy_extreme = [float('inf'), float('-inf')]
		self.review_scores_cleanliness_extreme = [float('inf'), float('-inf')]
		self.review_scores_checkin_extreme = [float('inf'), float('-inf')]
		self.review_scores_communication_extreme = [float('inf'), float('-inf')]
		self.review_scores_location_extreme = [float('inf'), float('-inf')]
		self.review_scores_value_extreme = [float('inf'), float('-inf')]
		self.reviews_per_month_extreme = [float('inf'), float('-inf')]
		self.total_room_nums_extreme = [float('inf'), float('-inf')]
		self.price_per_bedroom_extreme = [float('inf'), float('-inf')]
		self.description_X = []




	def self_train_test_split(self, X, y, ratio):
		n = len(y)

		num_val = int(n*ratio)

		X_val = []
		y_val = []
		for i in range(num_val):
			index = random.randint(0, len(y)-1)
			X_val.append(X.pop(index))
			y_val.append(y.pop(index))

		return X, y, X_val, y_val


	# output prediction for the test.csv
	def predict(self, X_df, ID):
		X_df = X_df.drop('name', axis=1)
		X_df = X_df.drop('description', axis=1)
		X_df = X_df.drop('bathrooms_text', axis=1)
		X_df = X_df.drop('reviews', axis=1)

		ID_list = ID.values.tolist()
		feature_names = X.columns.tolist()
		dim = len(feature_names)
		num = len(ID_list)
		X = X_df.values.tolist()
		X = self.convert2num(X, feature_names, [num, dim])

		for i in range(len(X)):
			X[i] = self.flatten(X[i])


		X = np.array(X).astype(np.float32)
		X = np.nan_to_num(X, nan=0)
		X = np.array(X).astype(np.float32)


		

		dataset = TensorDataset(torch.tensor(X).to(device))
		loader = DataLoader(dataset, batch_size=16, shuffle=False)

		self.mlp.to(device)

		Y_pre = []
		with torch.no_grad():
			self.mlp.eval()
			for inputs in test_loader:
				outputs = self.mlp(inputs)
				Y_pre = Y_pre + torch.round(outputs.squeeze()).cpu().numpy().tolist()

		prediction = {
			"id": ID_list,
			"price": Y_pre
		}

		output = pd.DataFrame(prediction)
		output.to_csv("output.csv", index=False)






	#flatten one hot encoded features so that each category is a new feature under the dataframe
	def flatten(self, lst):
	    flattened_list = []
	    for item in lst:
	        if isinstance(item, list):
	            flattened_list.extend(self.flatten(item))
	        else:
	            flattened_list.append(item)
	    return flattened_list






	def train(self, X_df, y_df):

		# we believe the following features does is irrelavant, for details please see the report
		X_df = X_df.drop('name', axis=1)
		X_df = X_df.drop('description', axis=1)
		X_df = X_df.drop('bathrooms_text', axis=1)
		X_df = X_df.drop('reviews', axis=1)

		# hyperparamters
		batch_size = 16
		reg = 'l1'
		epochs = 100
		lam_reg = 0.0001
		lr = 0.001
		num_hidden = 2

		feature_names = X_df.columns.tolist()
		dim = len(feature_names)

		X = X_df.values.tolist()
		y = y_df.values.tolist()


		num_points_total = len(y)


		X_train, y_train, X_val, y_val = self.self_train_test_split(X, y, 0.2)

		num_train_points = len(y_train)
		num_val_points = len(y_val)

		y_train = np.array(y_train).astype(np.float32)
		y_val = np.array(y_val).astype(np.float32)


		# record normalization paramters
		self.engineer_feature(X_train, feature_names, [num_train_points, dim])

		# normalize the features with recorded paramters
		X_train = self.convert2num(X_train, feature_names, [num_train_points, dim])
		X_val = self.convert2num(X_val, feature_names, [num_val_points, dim])

		# flatten the training data to a shape of (n,d)
		for i in range(len(X_train)):
			X_train[i] = self.flatten(X_train[i])

		for i in range(len(X_val)):
			X_val[i] = self.flatten(X_val[i])



		X_train = np.array(X_train).astype(np.float32)
		X_train = np.nan_to_num(X_train, nan=0)
		X_train = np.array(X_train).astype(np.float32)
		X_val = np.array(X_val).astype(np.float32)
		X_val = np.nan_to_num(X_val, nan=0)
		X_val = np.array(X_val).astype(np.float32)
		train_dataset = TensorDataset(torch.tensor(X_train).to(device), torch.tensor(y_train).to(device))
		val_dataset = TensorDataset(torch.tensor(X_val).to(device), torch.tensor(y_val).to(device))
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



		d = X_train.shape[1]
		n = X_train.shape[0]


		# number of perceptrons at each hidden layer
		hidden_size_list = []
		for i in range(num_hidden):
			hidden_size_list.append(int(1.5*d))

		

		self.mlp = MLP(input_size=d, hidden_size_list=hidden_size_list, output_size=1)
		self.mlp.to(device)

		criterion = nn.MSELoss()
		optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

		train_acc = []
		val_acc = []
		epoch_loss = []
		itr = 0
		itr_list = []
		for epoch in range(epochs):
			self.mlp.train()
			for inputs, targets in train_loader:
				optimizer.zero_grad()
				outputs = self.mlp(inputs)
				loss = criterion(outputs.squeeze(), targets) 

				# lasso regularization
				if reg == 'l1':
					penalty = sum((w.abs().sum()) for w in self.mlp.parameters())
					loss = loss + penalty * lam_reg


				
				loss.backward()
				optimizer.step()
			scheduler.step()
			itr_list.append(itr)
			itr = itr + 1
			acc_val = self.evaluate_model(val_loader)
			acc_train = self.evaluate_model(train_loader)
			val_acc.append(acc_val)
			train_acc.append(acc_train)
			print(f'Epoch: {itr}, Val Acc: {acc_val}, Train Acc: {acc_train}')


		plt.plot(itr_list, val_acc, label='validation')
		plt.plot(itr_list, train_acc, label='train')
		plt.legend()
		plt.show()

		return val_acc[-1]
		


	def evaluate_model(self, test_loader):
		self.mlp.to(device)
		
		correct = 0
		total = 0

		Y_pre = []
		Y = []

		with torch.no_grad():
			self.mlp.eval()
			for inputs, targets in test_loader:
				outputs = self.mlp(inputs)
				Y = Y + torch.round(targets).cpu().numpy().tolist()
				Y_pre = Y_pre + torch.round(outputs.squeeze()).cpu().numpy().tolist()

		return root_mean_squared_error(Y, Y_pre)


	# noramlize the data with previously recorded noramlization parameters
	def convert2num(self, feature, feature_names, shape):
		[n,d] = shape
		for i in range(n):
			for j in range(d):
				match feature_names[j]:
					case "property_type":
						x = np.zeros(len(self.property_type_set) + 1)
						not_found = True
						for k in range(len(self.property_type_set)):
							if feature[i][j] == self.property_type_set[k]:
								not_found = False
								x[k] = 1.0
						if not_found:
							x[-1] = 1.0
						feature[i][j] = x.tolist()
					case "neighbourhood_cleansed":
						x = np.zeros(len(self.neighbourhood_cleansed_set) + 1)
						not_found = True
						for k in range(len(self.neighbourhood_cleansed_set)):
							if feature[i][j] == self.neighbourhood_cleansed_set[k]:
								not_found = False
								x[k] = 1.0
						if not_found:
							x[-1] = 1.0
						feature[i][j] = x.tolist()

					case "neighbourhood_group_cleansed":
						x = np.zeros(len(self.neighbourhood_group_cleansed_set) + 1)
						not_found = True
						for k in range(len(self.neighbourhood_group_cleansed_set)):
							if feature[i][j] == self.neighbourhood_group_cleansed_set[k]:
								not_found = False
								x[k] = 1.0
						if not_found:
							x[-1] = 1.0
						feature[i][j] = x.tolist()

					case "latitude":
						feature[i][j] = (feature[i][j] - self.latitude_extreme[0]*1.0)/(self.latitude_extreme[1] - self.latitude_extreme[0])

					case "longitude":
						feature[i][j] = (feature[i][j] - self.longitude_extreme[0]*1.0)/(self.longitude_extreme[1] - self.longitude_extreme[0])

					case "host_since":
						feature[i][j] = (feature[i][j].timestamp() - self.host_since_extreme[0]*1.0)/(self.host_since_extreme[1] - self.host_since_extreme[0])

					case "host_response_time":
						x = np.zeros(len(self.host_response_time_set) + 1)
						not_found = True
						for k in range(len(self.host_response_time_set)):
							if feature[i][j] == self.host_response_time_set[k]:
								not_found = False
								x[k] = 1.0
						if not_found:
							x[-1] = 1.0
						feature[i][j] = x.tolist()

					case "host_response_rate":
						feature[i][j] = (feature[i][j] - self.host_response_rate_extreme[0]*1.0)/(self.host_response_rate_extreme[1] - self.host_response_rate_extreme[0])

					case "host_acceptance_rate":
						feature[i][j] = (feature[i][j] - self.host_acceptance_rate_extreme[0]*1.0)/(self.host_acceptance_rate_extreme[1] - self.host_acceptance_rate_extreme[0])

					case "host_is_superhost":
						if not math.isnan(feature[i][j]):
							feature[i][j] = float(int(feature[i][j]))
						else:
							feature[i][j] = -1.0

					case "host_listings_count":
						feature[i][j] = (feature[i][j] - self.host_listings_count_extreme[0]*1.0)/(self.host_listings_count_extreme[1] - self.host_listings_count_extreme[0])

					case "host_total_listings_count":
						feature[i][j] = (feature[i][j] - self.host_total_listings_count_extreme[0]*1.0)/(self.host_total_listings_count_extreme[1] - self.host_total_listings_count_extreme[0])

					case "host_verifications":
						x = np.zeros(len(self.host_verifications_set) + 1)
						not_found = True
						for k in range(len(self.host_verifications_set)):
							if feature[i][j] == self.host_verifications_set[k]:
								not_found = False
								x[k] = 1.0
						if not_found:
							x[-1] = 1.0
						feature[i][j] = x.tolist()

					case "host_has_profile_pic":
						feature[i][j] = float(int(feature[i][j]))

					case "host_identity_verified":
						feature[i][j] = float(int(feature[i][j]))

					case "calculated_host_listings_count":
						feature[i][j] = (feature[i][j] - self.calculated_host_listings_count_extreme[0]*1.0)/(self.calculated_host_listings_count_extreme[1] - self.calculated_host_listings_count_extreme[0])

					case "calculated_host_listings_count_entire_homes":
						feature[i][j] = (feature[i][j] - self.calculated_host_listings_count_entire_homes_extreme[0]*1.0)/(self.calculated_host_listings_count_entire_homes_extreme[1] - self.calculated_host_listings_count_entire_homes_extreme[0])

					case "calculated_host_listings_count_private_rooms":
						feature[i][j] = (feature[i][j] - self.calculated_host_listings_count_private_rooms_extreme[0]*1.0)/(self.calculated_host_listings_count_private_rooms_extreme[1] - self.calculated_host_listings_count_private_rooms_extreme[0])

					case "calculated_host_listings_count_shared_rooms":
						feature[i][j] = (feature[i][j] - self.calculated_host_listings_count_shared_rooms_extreme[0]*1.0)/(self.calculated_host_listings_count_shared_rooms_extreme[1] - self.calculated_host_listings_count_shared_rooms_extreme[0])

					case "room_type":
						x = np.zeros(len(self.room_type_set) + 1)
						not_found = True
						for k in range(len(self.room_type_set)):
							if feature[i][j] == self.room_type_set[k]:
								not_found = False
								x[k] = 1.0
						if not_found:
							x[-1] = 1.0
						feature[i][j] = x.tolist()

					case "accomodates":
						feature[i][j] = (feature[i][j] - self.accommodates_extreme[0]*1.0)/(self.accommodates_extreme[1] - self.accommodates_extreme[0])

					case "bathrooms":
						feature[i][j] = (feature[i][j] - self.bathrooms_extreme[0]*1.0)/(self.bathrooms_extreme[1] - self.bathrooms_extreme[0])

					case "bathrooms_text":
						feature[i][j] = np.random.randn()
					case "bedrooms":
						feature[i][j] = (feature[i][j] - self.bedrooms_extreme[0]*1.0)/(self.bedrooms_extreme[1] - self.bedrooms_extreme[0])

					case "beds":
						feature[i][j] = (feature[i][j] - self.beds_extreme[0]*1.0)/(self.beds_extreme[1] - self.beds_extreme[0])

					case "amenities":
						not_found = True
						x = np.zeros(len(self.amenities_set) + 1)
						a_list = feature[i][j].replace('[', '').replace(']', '').replace('"', '').split(', ')
						for item in a_list:
							for k in range(len(self.amenities_set)):
								if item.lower() == self.amenities_set[k]:
									x[k] = 1.0
									not_found = False
						if not_found:
							x[-1] = 1.0
						feature[i][j] = np.sum(x.tolist())

					case "has_availability":
						if not math.isnan(feature[i][j]):
							feature[i][j] = float(int(feature[i][j]))
						else:
							feature[i][j] = -1.0

					case "availability_30":
						feature[i][j] = (feature[i][j] - self.avail_30_extreme[0]*1.0)/(self.avail_30_extreme[1] - self.avail_30_extreme[0])

					case "availability_60":
						feature[i][j] = (feature[i][j] - self.avail_60_extreme[0]*1.0)/(self.avail_60_extreme[1] - self.avail_60_extreme[0])

					case "availability_90":
						feature[i][j] = (feature[i][j] - self.avail_90_extreme[0]*1.0)/(self.avail_90_extreme[1] - self.avail_90_extreme[0])

					case "availability_365":
						feature[i][j] = (feature[i][j] - self.avail_365_extreme[0]*1.0)/(self.avail_365_extreme[1] - self.avail_365_extreme[0])

					case "instant_bookable":
						feature[i][j] = float(int(feature[i][j]))

					case "minimum_nights":
						feature[i][j] = (feature[i][j] - self.minimum_nights_extreme[0]*1.0)/(self.minimum_nights_extreme[1] - self.minimum_nights_extreme[0])

					case "maximum_nights":
						feature[i][j] = (feature[i][j] - self.maximum_nights_extreme[0]*1.0)/(self.maximum_nights_extreme[1] - self.maximum_nights_extreme[0])

					case "number_of_reviews":
						feature[i][j] = (feature[i][j] - self.num_reviews_extreme[0]*1.0)/(self.num_reviews_extreme[1] - self.num_reviews_extreme[0])

					case "number_of_reviews_ltm":
						feature[i][j] = (feature[i][j] - self.num_reviews_ltm_extreme[0]*1.0)/(self.num_reviews_ltm_extreme[1] - self.num_reviews_ltm_extreme[0])

					case "number_of_reviews_l30d":
						feature[i][j] = (feature[i][j] - self.num_reviews_l30d_extreme[0]*1.0)/(self.num_reviews_l30d_extreme[1] - self.num_reviews_l30d_extreme[0])

					case "first_review":
						if pd.notna(feature[i][j]):
							feature[i][j] = (feature[i][j].timestamp() - self.first_review_extreme[0]*1.0)/(self.first_review_extreme[1] - self.first_review_extreme[0])
						else:
							feature[i][j] = -1.0

					case "last_review":
						if pd.notna(feature[i][j]):
							feature[i][j] = (feature[i][j].timestamp() - self.last_review_extreme[0]*1.0)/(self.last_review_extreme[1] - self.last_review_extreme[0])
						else:
							feature[i][j] = -1.0
					case "review_scores_rating":
						feature[i][j] = (feature[i][j] - self.review_scores_rating_extreme[0]*1.0)/(self.review_scores_rating_extreme[1] - self.review_scores_rating_extreme[0])

					case "review_scores_accuracy":
						feature[i][j] = (feature[i][j] - self.review_scores_accuracy_extreme[0]*1.0)/(self.review_scores_accuracy_extreme[1] - self.review_scores_accuracy_extreme[0])

					case "review_scores_cleanliness":
						feature[i][j] = (feature[i][j] - self.review_scores_cleanliness_extreme[0]*1.0)/(self.review_scores_cleanliness_extreme[1] - self.review_scores_cleanliness_extreme[0])

					case "review_scores_checkin":
						feature[i][j] = (feature[i][j] - self.review_scores_checkin_extreme[0]*1.0)/(self.review_scores_checkin_extreme[1] - self.review_scores_checkin_extreme[0])

					case "review_scores_communication":
						feature[i][j] = (feature[i][j] - self.review_scores_communication_extreme[0]*1.0)/(self.review_scores_communication_extreme[1] - self.review_scores_communication_extreme[0])

					case "review_scores_location":
						feature[i][j] = (feature[i][j] - self.review_scores_location_extreme[0]*1.0)/(self.review_scores_location_extreme[1] - self.review_scores_location_extreme[0])

					case "review_scores_value":
						feature[i][j] = (feature[i][j] - self.review_scores_value_extreme[0]*1.0)/(self.review_scores_value_extreme[1] - self.review_scores_value_extreme[0])

					case "reviews_per_month":
						feature[i][j] = (feature[i][j] - self.reviews_per_month_extreme[0]*1.0)/(self.reviews_per_month_extreme[1] - self.reviews_per_month_extreme[0])

					case "total_room_nums":
						feature[i][j] = (feature[i][j] - self.total_room_nums_extreme[0]*1.0)/(self.total_room_nums_extreme[1] - self.total_room_nums_extreme[0])

					case "price_per_bedroom":
						feature[i][j] = (feature[i][j] - self.price_per_bedroom_extreme[0]*1.0)/(self.price_per_bedroom_extreme[1] - self.price_per_bedroom_extreme[0])

		return feature



	# record noramlization paramters
	def engineer_feature(self, feature, feature_names, shape):
		[n,d] = shape

		for i in range(n):
			for j in range(d):
				match feature_names[j]:

					case "property_type":
						self.property_type_set.add(feature[i][j])
					case "neighbourhood_cleansed":
						self.neighbourhood_cleansed_set.add(feature[i][j])
					case "neighbourhood_group_cleansed":
						self.neighbourhood_group_cleansed_set.add(feature[i][j])
					case "latitude":
						self.latitude_extreme[0] = min(self.latitude_extreme[0], feature[i][j])
						self.latitude_extreme[1] = max(self.latitude_extreme[1], feature[i][j])
					case "longitude":
						self.longitude_extreme[0] = min(self.longitude_extreme[0], feature[i][j])
						self.longitude_extreme[1] = max(self.longitude_extreme[1], feature[i][j])
					case "host_since":
						try:
							self.host_since_extreme[0] = min(self.host_since_extreme[0], feature[i][j].timestamp())
							self.host_since_extreme[1] = max(self.host_since_extreme[1], feature[i][j].timestamp())
						except:
							pass
					case "host_response_time":
						self.host_response_time_set.add(feature[i][j])
					case "host_response_rate":
						self.host_response_rate_extreme[0] = min(self.host_response_rate_extreme[0], feature[i][j])
						self.host_response_rate_extreme[1] = max(self.host_response_rate_extreme[1], feature[i][j])
					case "host_acceptance_rate":
						self.host_acceptance_rate_extreme[0] = min(self.host_acceptance_rate_extreme[0], feature[i][j])
						self.host_acceptance_rate_extreme[1] = max(self.host_acceptance_rate_extreme[1], feature[i][j])
					case "host_is_superhost":
						pass
					case "host_listings_count":
						self.host_listings_count_extreme[0] = min(self.host_listings_count_extreme[0], feature[i][j])
						self.host_listings_count_extreme[1] = max(self.host_listings_count_extreme[1], feature[i][j])
					case "host_total_listings_count":
						self.host_total_listings_count_extreme[0] = min(self.host_total_listings_count_extreme[0], feature[i][j])
						self.host_total_listings_count_extreme[1] = max(self.host_total_listings_count_extreme[1], feature[i][j])
					case "host_verifications":
						for item in feature[i][j]:
							self.host_verifications_set.add(item)
					case "host_has_profile_pic":
						pass
					case "host_identity_verified":
						pass
					case "calculated_host_listings_count":
						self.calculated_host_listings_count_extreme[0] = min(self.calculated_host_listings_count_extreme[0], feature[i][j])
						self.calculated_host_listings_count_extreme[1] = max(self.calculated_host_listings_count_extreme[1], feature[i][j])
					case "calculated_host_listings_count_entire_homes":
						self.calculated_host_listings_count_entire_homes_extreme[0] = min(self.calculated_host_listings_count_entire_homes_extreme[0], feature[i][j])
						self.calculated_host_listings_count_entire_homes_extreme[1] = max(self.calculated_host_listings_count_entire_homes_extreme[1], feature[i][j])
					case "calculated_host_listings_count_private_rooms":
						self.calculated_host_listings_count_private_rooms_extreme[0] = min(self.calculated_host_listings_count_private_rooms_extreme[0], feature[i][j])
						self.calculated_host_listings_count_private_rooms_extreme[1] = max(self.calculated_host_listings_count_private_rooms_extreme[1], feature[i][j])
					case "calculated_host_listings_count_shared_rooms":
						self.calculated_host_listings_count_shared_rooms_extreme[0] = min(self.calculated_host_listings_count_shared_rooms_extreme[0], feature[i][j])
						self.calculated_host_listings_count_shared_rooms_extreme[1] = max(self.calculated_host_listings_count_shared_rooms_extreme[1], feature[i][j])
					case "room_type":
						self.room_type_set.add(feature[i][j])
					case "accomodates":
						self.accommodates_extreme[0] = min(self.accommodates_extreme[0], feature[i][j])
						self.accommodates_extreme[1] = max(self.accommodates_extreme[1], feature[i][j])
					case "bathrooms":
						self.bathrooms_extreme[0] = min(self.bathrooms_extreme[0], feature[i][j])
						self.bathrooms_extreme[1] = max(self.bathrooms_extreme[1], feature[i][j])
					case "bathrooms_text":
						pass
					case "bedrooms":
						self.bedrooms_extreme[0] = min(self.bedrooms_extreme[0], feature[i][j])
						self.bedrooms_extreme[1] = max(self.bedrooms_extreme[1], feature[i][j])
					case "beds":
						self.beds_extreme[0] = min(self.beds_extreme[0], feature[i][j])
						self.beds_extreme[1] = max(self.beds_extreme[1], feature[i][j])
					case "amenities":
						a_list = feature[i][j].replace('[', '').replace(']', '').replace('"', '').split(', ')
						for item in a_list:
							self.amenities_set.add(item.lower())
					case "has_availability":
						pass
					case "availability_30":
						self.avail_30_extreme[0] = min(self.avail_30_extreme[0], feature[i][j])
						self.avail_30_extreme[1] = max(self.avail_30_extreme[1], feature[i][j])
					case "availability_60":
						self.avail_60_extreme[0] = min(self.avail_60_extreme[0], feature[i][j])
						self.avail_60_extreme[1] = max(self.avail_60_extreme[1], feature[i][j])
					case "availability_90":
						self.avail_90_extreme[0] = min(self.avail_90_extreme[0], feature[i][j])
						self.avail_90_extreme[1] = max(self.avail_90_extreme[1], feature[i][j])
					case "availability_365":
						self.avail_365_extreme[0] = min(self.avail_365_extreme[0], feature[i][j])
						self.avail_365_extreme[1] = max(self.avail_365_extreme[1], feature[i][j])
					case "instant_bookable":
						pass
					case "minimum_nights":
						self.minimum_nights_extreme[0] = min(self.minimum_nights_extreme[0], feature[i][j])
						self.minimum_nights_extreme[1] = max(self.minimum_nights_extreme[1], feature[i][j])
					case "maximum_nights":
						self.maximum_nights_extreme[0] = min(self.maximum_nights_extreme[0], feature[i][j])
						self.maximum_nights_extreme[1] = max(self.maximum_nights_extreme[1], feature[i][j])
					case "number_of_reviews":
						self.num_reviews_extreme[0] = min(self.num_reviews_extreme[0], feature[i][j])
						self.num_reviews_extreme[1] = max(self.num_reviews_extreme[1], feature[i][j])
					case "number_of_reviews_ltm":
						self.num_reviews_ltm_extreme[0] = min(self.num_reviews_ltm_extreme[0], feature[i][j])
						self.num_reviews_ltm_extreme[1] = max(self.num_reviews_ltm_extreme[1], feature[i][j])
					case "number_of_reviews_l30d":
						self.num_reviews_l30d_extreme[0] = min(self.num_reviews_l30d_extreme[0], feature[i][j])
						self.num_reviews_l30d_extreme[1] = max(self.num_reviews_l30d_extreme[1], feature[i][j])
					case "first_review":
						if pd.notna(feature[i][j]):
							self.first_review_extreme[0] = min(self.first_review_extreme[0], feature[i][j].timestamp())
							self.first_review_extreme[1] = max(self.first_review_extreme[1], feature[i][j].timestamp())
					case "last_review":
						if pd.notna(feature[i][j]):
							self.last_review_extreme[0] = min(self.last_review_extreme[0], feature[i][j].timestamp())
							self.last_review_extreme[1] = max(self.last_review_extreme[1], feature[i][j].timestamp())
					case "review_scores_rating":
						self.review_scores_rating_extreme[0] = min(self.review_scores_rating_extreme[0], feature[i][j])
						self.review_scores_rating_extreme[1] = max(self.review_scores_rating_extreme[1], feature[i][j])
					case "review_scores_accuracy":
						self.review_scores_accuracy_extreme[0] = min(self.review_scores_accuracy_extreme[0], feature[i][j])
						self.review_scores_accuracy_extreme[1] = max(self.review_scores_accuracy_extreme[1], feature[i][j])
					case "review_scores_cleanliness":
						self.review_scores_cleanliness_extreme[0] = min(self.review_scores_cleanliness_extreme[0], feature[i][j])
						self.review_scores_cleanliness_extreme[1] = max(self.review_scores_cleanliness_extreme[1], feature[i][j])
					case "review_scores_checkin":
						self.review_scores_checkin_extreme[0] = min(self.review_scores_checkin_extreme[0], feature[i][j])
						self.review_scores_checkin_extreme[1] = max(self.review_scores_checkin_extreme[1], feature[i][j])
					case "review_scores_communication":
						self.review_scores_communication_extreme[0] = min(self.review_scores_communication_extreme[0], feature[i][j])
						self.review_scores_communication_extreme[1] = max(self.review_scores_communication_extreme[1], feature[i][j])
					case "review_scores_location":
						self.review_scores_location_extreme[0] = min(self.review_scores_location_extreme[0], feature[i][j])
						self.review_scores_location_extreme[1] = max(self.review_scores_location_extreme[1], feature[i][j])
					case "review_scores_value":
						self.review_scores_value_extreme[0] = min(self.review_scores_value_extreme[0], feature[i][j])
						self.review_scores_value_extreme[1] = max(self.review_scores_value_extreme[1], feature[i][j])
					case "reviews_per_month":
						self.reviews_per_month_extreme[0] = min(self.reviews_per_month_extreme[0], feature[i][j])
						self.reviews_per_month_extreme[1] = max(self.reviews_per_month_extreme[1], feature[i][j])
					case "total_room_nums":
						self.total_room_nums_extreme[0] = min(self.total_room_nums_extreme[0], feature[i][j])
						self.total_room_nums_extreme[1] = max(self.total_room_nums_extreme[1], feature[i][j])
					case "price_per_bedroom":
						self.price_per_bedroom_extreme[0] = min(self.price_per_bedroom_extreme[0], feature[i][j])
						self.price_per_bedroom_extreme[1] = max(self.price_per_bedroom_extreme[1], feature[i][j])
					case "reviews":
						pass

		self.set2list()

	# convert the set to list so that we may iterate for further use
	def set2list(self):
		self.property_type_set = list(self.property_type_set)
		self.neighbourhood_cleansed_set = list(self.neighbourhood_cleansed_set)
		self.neighbourhood_group_cleansed_set = list(self.neighbourhood_group_cleansed_set)
		self.host_verifications_set = list(self.host_verifications_set)
		self.room_type_set = list(self.room_type_set)
		self.amenities_set = list(self.amenities_set)
		self.host_response_time_set = list(self.host_response_time_set)