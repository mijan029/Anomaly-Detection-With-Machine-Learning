#                                   THESIS PROJECT - CONVOLUTIONAL NEURAL NETWORK
#                                     Writer: Mijanur Rahman, Md. Faisal Zaman
#               -------------------------------------------------------------------------------------------
#               -------------------------------------------------------------------------------------------

# sir k request kor j amra robibar joma dibo.......shoni r robi kaj korbo cnn niye

#CNN layers
#Input layer -> Convolutional layer -> Activation layer -> Pooling layer -> Flattening -> Fully Connected layers -> Output layer

#Completed
# - Input layer
# - Convolutional layer
# - Activation layer
# - Pooling layer
# - Flattening layer
# - Fully Connected layer
# - Output layer


#TO-DO  List
# - Iteration increasing -- See Documentation & Video Tutorial
# - Linear Transform  -> Softmax

#This is input layer.
import numpy as np
import keras
from keras.datasets import cifar10

# Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_shape = 1000
x_train = x_train[0:x_train_shape]
y_train = y_train[0:x_train_shape]

# Normalizing pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshaping the labels to be one-dimensional arrays
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)


image_height, image_width, num_channels = x_train[0].shape
print(x_train[0])
print(image_height, image_width, num_channels)

#convolutional layer
def conv2d(image, kernel):
    # Getting dimensions
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width, _ = kernel.shape

    # Calculating output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initializing output feature map
    output = np.zeros((output_height, output_width))

    # Performing convolution
    for i in range(output_height):
        for j in range(output_width):
            for c in range(num_channels):
                output[i, j] += np.sum(image[i:i+kernel_height, j:j+kernel_width, c] * kernel[:, :, c])

    return output


filter_size = 3
num_channels = 3

# Creating a random filter
kernel = np.random.randn(filter_size, filter_size, num_channels)
print(kernel)
# filter_biases = np.random.randn(num_filters)

x_convolutional = [np.zeros((30, 30)) for _ in range(x_train_shape)]
for i in range(x_train_shape):
    x_convolutional[i] = conv2d(x_train[i],kernel)



# Activation layer
def activation_layer(input_data):
    return np.maximum(0, input_data)

for i in range(x_train_shape):
    x_convolutional[i] = activation_layer(x_convolutional[i])
#print(x_convolutional[0])

x_activation = x_convolutional



#Pooling layer

def pooling_layer(input_data, pool_size=2, pool_type='max'):
    input_height = len(input_data)
    input_width = len(input_data[0])
    pool_height, pool_width = pool_size, pool_size
    

    # Calculating output dimensions
    foutput_height = input_height // pool_height
    foutput_width = input_width // pool_width
    output_height=int(foutput_height)
    output_width=int(foutput_width)

    # Initializing output feature map
    output = np.zeros((output_height, output_width))

    # Performing pooling
    for i in range(output_height):
        for j in range(output_width):
                if pool_type == 'max':
                    mm = 0.0;
                    for l in range(pool_height):
                        for k in range(pool_width):
                            mm = max(mm, input_data[i*pool_height+l][j*pool_width+k])
                    output[i, j] = mm
              

    return output

#x_pooling = np.zeros((15, 15))
for i in range(x_train_shape):
    x_activation[i] = pooling_layer(x_activation[i])
x_pooling = x_activation
    
print(x_pooling[0])
    



#Flattening
def flatten(input_data):
    flattened_data = input_data.flatten()
    return flattened_data

for i in range(x_train_shape):
    x_pooling[i] =flatten(x_pooling[i])
x_flatten = x_pooling

weights= np.zeros((225, 225))
for i in range (225):
    for j in range (225):
        weights[i][j]=i+1;
biases=np.zeros((1, 225))

# Fully Connected Component
def fully_connected_layer(input_data, weights, biases):
    # Performing matrix multiplication
    output = np.dot(input_data, weights) + biases
    output = np.maximum(0, output)
    return output

for i in range(x_train_shape):
    x_flatten[i] =fully_connected_layer(x_flatten[i],weights,biases)
x_fully_connected_layer=x_flatten

#Output Layer
def output_layer(input_data, weights, biases): # activation_function= linear
    # Performing matrix multiplication
    output = np.dot(input_data, weights) + biases
    
    return output
for i in range(x_train_shape):
    x_fully_connected_layer[i] =output_layer(x_fully_connected_layer[i],weights,biases)
x_output=x_fully_connected_layer

