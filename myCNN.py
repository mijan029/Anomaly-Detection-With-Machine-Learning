#                                   THESIS PROJECT - CONVOLUTIONAL NEURAL NETWORK
#                                     Writer: Mijanur Rahman, Md. Faisal Zaman
#               -------------------------------------------------------------------------------------------
#               -------------------------------------------------------------------------------------------


#CNN layers
# Input layer -> Convolutional layer -> Activation layer -> Pooling layer -> Flattening -> Fully Connected layers -> Output layer

#Completed
# - Input layer
# - Convolutional layer
# - Activation layer
# - Pooling layer
# - Flattening layer
# - Fully Connected layer
# - 


#TO-DO  List
# - Iteration increasing -- See Documentation & Video Tutorial
# - Linear Transform  -> Softmax

#This is input layer.
import numpy as np
import keras
from keras.datasets import cifar10

# Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_shape = 100
iteration_number = 10
x_train = x_train[0:x_train_shape]
y_train = y_train[0:x_train_shape]


# Normalizing pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshaping the labels to be one-dimensional arrays
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

# print(y_train)

image_height, image_width, num_channels = x_train[0].shape
# print(x_train[0])
# print(image_height, image_width, num_channels)

j=0
#convolutional layer
def conv2d(image, kernel):
    # Getting dimensions
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width, kernel_channel = kernel.shape

    # Calculating output dimensions
    
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initializing output feature map
    output = np.zeros((output_height, output_width, kernel_channel))

    # Performing convolution
    for i in range(output_height):
        for j in range(output_width):
                output[i,j,:] = np.sum(image[i:i+kernel_height, j:j+kernel_width, :] * kernel[:, :, :])

    return output


filter_size = 3
num_channels = 3

# Creating a random filter
kernel = np.random.randn(filter_size, filter_size, num_channels)
# print(kernel)
# filter_biases = np.random.randn(num_filters)

x_convolutional = [np.zeros((32, 32, 3)) for _ in range(x_train_shape)]
for i in range(x_train_shape):
    for j in range (iteration_number):
        if(j==0):
            x_convolutional[i] = conv2d(x_train[i],kernel)
        else:
            x_convolutional[i] = conv2d(x_convolutional[i],kernel)

# x_convolutional_final=[np.zeros((32-2*iteration_number, 32-2*iteration_number)) for _ in range(x_train_shape)]
# row=32-2*iteration_number
# column=32-2*iteration_number
# for i in range(x_train_shape):
#     for j in range(row):
#         for k in range(column):
#             x_convolutional_final[i][j][k]=x_convolutional[i][j][k]

# Activation layer


def activation_layer(input_data):
    return np.maximum(0, input_data)

for i in range(x_train_shape):
    x_convolutional[i] = activation_layer(x_convolutional[i])
#print(x_convolutional[0])

x_activation = x_convolutional


# merging all channels into one
x_activation = np.sum(x_activation,axis=3)
# print(x_activation.shape)

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
                    mm = 0.0
                    for l in range(pool_height):
                        for k in range(pool_width):
                            mm = max(mm, input_data[i*pool_height+l,j*pool_width+k])
                    output[i, j] = mm
                    
    
              

    return output

#x_pooling = np.zeros((15, 15))
x_pooling = [np.zeros((12, 12)) for _ in range(x_train_shape)]
for i in range(x_train_shape):
    x_pooling[i] = pooling_layer(x_activation[i])
    
# print(x_pooling[0].shape)
    



#Flattening
#After flattening the array has 144 elements
x_flatten  = [np.zeros((36)) for _ in range(x_train_shape)]
def flatten(input_data):
    flattened_data = input_data.flatten()
    return flattened_data

for i in range(x_train_shape):
    x_flatten[i] = flatten(x_pooling[i])


weight1= np.zeros((36, 15))
for i in range (36):
    for j in range (15):
        weight1[i][j]=i+1
biase1=np.random.rand(15)

weight2= np.zeros((15, 9))
for i in range (15):
    for j in range (9):
        weight2[i][j]=i+1
biase2=np.random.rand(9)

weight3= np.zeros((9, 4))
for i in range (9):
    for j in range (4):
        weight3[i][j]=i+1
biase3=np.random.rand(4)

# Fully Connected Component
def fully_connected_layer(input_data, weights, biases):
    # Performing matrix multiplication
    output = np.dot(input_data, weights) + biases
    output = np.maximum(0, output)
    return output

x_fully_connected_layer_1=[np.zeros((1, 15))for _ in range(x_train_shape)]
x_fully_connected_layer_2=[np.zeros((1, 9))for _ in range(x_train_shape)]
x_fully_connected_layer_final=[np.zeros((1, 4))for _ in range(x_train_shape)]
for i in range(x_train_shape):
    x_fully_connected_layer_1[i] = fully_connected_layer(x_flatten[i],weight1,biase1)
for i in range(x_train_shape):
    x_fully_connected_layer_2[i] =fully_connected_layer(x_fully_connected_layer_1[i],weight2,biase2)
for i in range(x_train_shape):
    x_fully_connected_layer_final[i] =fully_connected_layer(x_fully_connected_layer_2[i],weight3,biase3)


# print(x_fully_connected_layer_final)



# Output Layer
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)


def output_layer(input_data): # activation_function= softmax
    output = softmax(input_data)
    return output

# the size of fully connected layer is 4
Output  = [np.zeros((4)) for _ in range(x_train_shape)]
Output = output_layer(x_fully_connected_layer_final)


print(Output)









