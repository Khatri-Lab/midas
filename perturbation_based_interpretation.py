 from generate_data_train_models import *
 
# Generate example data
# Contains 4 columns: v0, v1, v2, and v3
# All columns are scaled to N(0, 1)
training_data = generate_tabular_data(seed = 11)
test_data = generate_tabular_data(seed = 101)

# Train models in round-robin manner
nn_models_list = []
for output_column in range(4):
  input_columns = [0, 1, 2, 3]
  input_columns.remove(output_column)
  nn_models_list.append(trained_model(training_data, input_columns, output_column))
# Models are fixed from hereon

# Compute R2 on test data
test_R2_list = []
for output_column in range(4):
  input_columns = [0, 1, 2, 3]
  input_columns.remove(output_column)
  test_R2_list.append(compute_R2(nn_models_list[output_column], test_data, input_columns, output_column))

# Perturb each input variable systematically for a given output variable
# Compute association strength as relative decrease in R2
association_strength_matrix = [[None] * 4 for i in range(4)]
for output_column in range(4):
  input_columns = [0, 1, 2, 3]
  input_columns.remove(output_column)
  for perturb_column in input_columns:
    test_data_perturbed = test_data.copy()
    test_data_perturbed[:, perturb_column] = generate_random_noise(seed = 1001)
    perturb_R2 = compute_R2(nn_models_list[output_column], test_data_perturbed, input_columns, output_column)
    association_strength_value = ((test_R2_list[output_column] - perturb_R2)/test_R2_list[output_column]) * 100
    association_strength_matrix[perturb_column][output_column] = association_strength_value
