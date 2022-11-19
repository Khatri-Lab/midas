import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

num_points = 10000

def generate_tabular_data(seed):
  np.random.seed(seed)
  v0 = np.random.normal(size = num_points)
  np.random.seed(seed + 1)
  v1 = np.random.normal(size = num_points)
  np.random.seed(seed + 2)
  #v2 = np.random.normal(size = num_points)
  v2 = np.array([max(i, 0) for i in v0])  + np.random.normal(size = num_points) * 0.005
  np.random.seed(seed + 3)
  v3 = 5 * v0 + 2 * v1 + v2 + np.random.normal(size = num_points) * 0.005
  v3 = (v3 - np.mean(v3))/np.std(v3)
  tabular_data = np.column_stack((v0, v1, v2, v3))
  return tabular_data

def trained_model(data_array, input_columns, output_column):
  nn_model = MLPRegressor(hidden_layer_sizes=(3),
    alpha = 0, learning_rate_init = 0.01, random_state=1)
  nn_model.fit(data_array[:, input_columns], data_array[:, output_column])
  return nn_model

def compute_R2(nn_model, data_array, input_columns, output_column):
  nn_prediction = nn_model.predict(data_array[:, input_columns])
  return round(max(r2_score(data_array[:, output_column], nn_prediction), 0), 3)

def generate_random_noise(seed):
  np.random.seed(seed)
  return np.random.normal(size = num_points)
