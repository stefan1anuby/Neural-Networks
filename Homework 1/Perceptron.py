import numpy as np
import matplotlib.pyplot as plt
import sys

from Activations import softmax,sigmoid

np.set_printoptions(threshold=sys.maxsize)

class Perceptron():
	
	def __init__(self, learning_rate=0.01 , n_batches = 4 , n_neurons_output = 10 , activation_function = softmax) -> None:
		self.learning_rate = learning_rate
		self.n_batches = n_batches
		self.n_neurons_output = n_neurons_output
		self.activation_function = activation_function

	def fit(self, train_set):
		
		self.w = np.zeros(shape = (self.n_neurons_output , len(train_set[0][0]) ) )
		self.b = np.zeros(self.n_neurons_output)
		
		batches_images , batches_labels = self.__split(train_set, self.n_batches)
		cnt = 0
		for batch_images , batch_labels in zip(batches_images,batches_labels):

			w_temp , b_temp = self.__train(batch_images , batch_labels)
			self.w = self.w + w_temp
			self.b = self.b + b_temp
			
			cnt += 1
			accuracy = self.compute_accuracy(train_set) * 100
			print(f"Batch {cnt}/{self.n_batches} accuracy is : {accuracy:.4f}%")


	def compute_accuracy(self, sample_set):
		
		sample_set_images , sample_set_labels = sample_set
		total = len(sample_set_images)
		wrong = 0
		for sample_image , sample_label in zip(sample_set_images,sample_set_labels):
			prediction = self.predict(sample_image)
			
			#print(f"prediction : {prediction}")
			#print(f"truth	   : {sample_label}")
			
			if np.argmax(prediction) != np.argmax(sample_label):
				wrong += 1

		return (total - wrong) / total
		

	def __train(self,train_set_images , train_set_labels):

		w = np.zeros(shape = (self.n_neurons_output , len(train_set_images[0])))
		b = np.zeros(self.n_neurons_output)

		linear_output = np.dot(train_set_images, w.T) + b
		probabilities = self.activation_function(linear_output)

		error = (probabilities - train_set_labels)

		num = train_set_images.shape[0]
		gradient = np.dot(error.T , train_set_images) / num
		bias_gradient = np.sum(error, axis=0) / num
		
		w -= self.learning_rate * gradient
		b -= self.learning_rate * bias_gradient

		return (w , b)

	def __split(self, sampleset , number):
		images , labels = sampleset
		
		# shuffle the data
		shuffled_indices = np.arange(len(images))
		np.random.shuffle(shuffled_indices)

		new_images = images[shuffled_indices]
		new_labels = labels[shuffled_indices]
		
		# split the data
		batches_images = np.array_split(new_images , number)
		batches_labels = np.array_split(new_labels , number)

		"""
		# DEBUGGING PURPOSES
		index = 0 
		batch = 0
		#show_image_and_label(new_images[index] , new_labels[index])
		show_image_and_label(batches_images[batch][index],batches_labels[batch][index])

		"""
		return (batches_images , batches_labels)
	
	def predict(self , X):
		net_input = np.dot(self.w , X) + self.b
		output = self.activation_function(net_input)
		y_predicted = output
		return y_predicted 
	

def show_image_and_label(image , label):
	plt.imshow(image.reshape(28, 28))
	title_obj = plt.title(str(label)) 
	plt.setp(title_obj)
	plt.show()



	


	