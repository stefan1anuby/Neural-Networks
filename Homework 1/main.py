import pickle, gzip
import numpy as np
from Perceptron import Perceptron
from Activations import softmax , sigmoid

with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")	

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x , test_y = test_set

print(f"train_x.shape = {train_x.shape} , valid_x.shape = {valid_x.shape} ,  test_x.shape = {test_x.shape}")
print(f"train_y.shape = {train_y.shape} , valid_y.shape = {valid_y.shape} ,  test_y.shape = {test_y.shape}")

num_classes = 10

train_y_onehot = np.eye(num_classes)[train_y]
valid_y_onehot = np.eye(num_classes)[valid_y]
test_y_onehot  = np.eye(num_classes)[test_y]

new_train_set = (train_x , train_y_onehot)
new_valid_set = (valid_x , valid_y_onehot)
new_test_set  = (test_x  , test_y_onehot)

combined_train_set = (
    np.concatenate((train_x, test_x), axis=0) ,
    np.concatenate((train_y_onehot, test_y_onehot), axis=0)
)

model = Perceptron(n_batches=10,learning_rate=0.1 , activation_function= softmax)
#model.fit(new_train_set)
model.fit(combined_train_set)

valid_set_accuracy = model.compute_accuracy(new_valid_set) * 100
print(f"Accuracy for validation set is : {valid_set_accuracy:.4}%")