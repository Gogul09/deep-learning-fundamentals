#-------------------
# organize imports
#-------------------
import numpy as np
import os
import h5py
import glob
import cv2
from keras.preprocessing import image

#------------------------
# dataset pre-processing
#------------------------
train_path   = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\train"
test_path    = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test"
train_labels = os.listdir(train_path)
test_labels  = os.listdir(test_path) 

# tunable parameters
image_size       = (64, 64)
num_train_images = 1500
num_test_images  = 101
num_channels     = 3
epochs           = 2000
lr               = 0.01

# train_x dimension = {(64*64*3), 1400}
# train_y dimension = {1, 1400}
# test_x dimension  = {(64*64*3), 200}
# test_y dimension  = {1, 200}
train_x = np.zeros(((image_size[0]*image_size[1]*num_channels), num_train_images))
train_y = np.zeros((1, num_train_images))
test_x  = np.zeros(((image_size[0]*image_size[1]*num_channels), num_test_images))
test_y  = np.zeros((1, num_test_images))

#----------------
# TRAIN dataset
#----------------
count = 0
num_label = 0
for i, label in enumerate(train_labels):
	cur_path = train_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x   = image.img_to_array(img)
		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)
		train_x[:,count] = x
		train_y[:,count] = num_label
		count += 1
	num_label += 1

#--------------
# TEST dataset
#--------------
count = 0 
num_label = 0 
for i, label in enumerate(test_labels):
	cur_path = test_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x   = image.img_to_array(img)
		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)
		test_x[:,count] = x
		test_y[:,count] = num_label
		count += 1
	num_label += 1

print ("train_labels : " + str(train_labels))
print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x shape : " + str(test_x.shape))
print ("test_y shape : " + str(test_y.shape))

#------------------
# standardization
#------------------
train_x = train_x/255.
test_x  = test_x/255.

#-----------------
# save using h5py
#-----------------
h5_train = h5py.File("train_x.h5", 'w')
h5_train.create_dataset("data_train", data=np.array(train_x))
h5_train.close()

h5_test = h5py.File("test_x.h5", 'w')
h5_test.create_dataset("data_test", data=np.array(test_x))
h5_test.close()

#----------------
# define sigmoid
#----------------
def sigmoid(z):
	return (1/(1+np.exp(-z)))

#---------------------------
# parameter initialization
#---------------------------
def init_params(dimension):
	w = np.zeros((dimension, 1))
	b = 0
	return w, b

#------------------------------
# forward and back propagation
#------------------------------
def propagate(w, b, X, Y):
	# num of training samples
	m = X.shape[1]

	# forward pass
	A    = sigmoid(np.dot(w.T,X) + b)
	cost = (-1/m)*(np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A))))

	# back propagation
	dw = (1/m)*(np.dot(X, (A-Y).T))
	db = (1/m)*(np.sum(A-Y))

	cost = np.squeeze(cost)

	# gradient dictionary
	grads = {"dw": dw, "db": db}

	return grads, cost

#------------------
# gradient descent
#------------------
def optimize(w, b, X, Y, epochs, lr):
	costs = []
	for i in range(epochs):
		# calculate gradients
		grads, cost = propagate(w, b, X, Y)

		# get gradients
		dw = grads["dw"]
		db = grads["db"]

		# update rule
		w = w - (lr*dw)
		b = b - (lr*db)

		if i % 100 == 0:
			costs.append(cost)
			print ("cost after %i epochs: %f" %(i, cost))

	# param dict
	params = {"w": w, "b": b}

	# gradient dict
	grads  = {"dw": dw, "db": db}

	return params, grads, costs

#-------------------------------------
# make prediction on multiple images
#-------------------------------------
def predict(w, b, X):
	m = X.shape[1]
	Y_predict = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(np.dot(w.T, X) + b)

	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_predict[0, i] = 0
		else:
			Y_predict[0,i]  = 1

	return Y_predict

#-------------------------------------
# make prediction on a single image
#-------------------------------------
def predict_image(w, b, X):
	Y_predict = None
	w = w.reshape(X.shape[0], 1)
	A = sigmoid(np.dot(w.T, X) + b)
	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_predict = 0
		else:
			Y_predict = 1

	return Y_predict

#----------------------
# logistic regression
#----------------------
def model(X_train, Y_train, X_test, Y_test, epochs, lr):
	w, b = init_params(X_train.shape[0])
	params, grads, costs = optimize(w, b, X_train, Y_train, epochs, lr)

	w = params["w"]
	b = params["b"]

	Y_predict_train = predict(w, b, X_train)
	Y_predict_test  = predict(w, b, X_test)

	print ("train_accuracy: {} %".format(100-np.mean(np.abs(Y_predict_train - Y_train)) * 100))
	print ("test_accuracy : {} %".format(100-np.mean(np.abs(Y_predict_test  - Y_test)) * 100))

	lr_model = {"costs": costs,
				"Y_predict_test": Y_predict_test, 
				"Y_predict_train" : Y_predict_train, 
				"w" : w, 
				"b" : b,
				"learning_rate" : lr,
				"epochs": epochs}

	return lr_model

# activate the logistic regression model
myModel = model(train_x, train_y, test_x, test_y, epochs, lr)

#------------------------------
# test images using our model
#------------------------------
test_img_paths = ["G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0723.jpg",
				  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0713.jpg",
				  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0782.jpg",
				  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0799.jpg",
				  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\test_1.jpg"]

for test_img_path in test_img_paths:
	img_to_show 	= cv2.imread(test_img_path, -1)
	img         	= image.load_img(test_img_path, target_size=image_size)
	x           	= image.img_to_array(img)
	x           	= x.flatten()
	x   			= np.expand_dims(x, axis=1)
	predict     	= predict_image(myModel["w"], myModel["b"], x)
	predict_label 	= ""

	if predict == 0:
		predict_label = "airplane"
	else:
		predict_label = "bike"

	# display the test image and the predicted label
	cv2.putText(img_to_show, predict_label, (30,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("test_image", img_to_show)
	key = cv2.waitKey(0) & 0xFF
	if (key == 27):
		cv2.destroyAllWindows()