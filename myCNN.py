from PIL import Image
import numpy as np

#CNN layer
#Input layer -> Convolutional layer -> Activation layer -> Pooling layer -> Flattering -> Fully Connected layers -> Output layer

#Completed
# - Input layer
# - Convolutional layer
# - Activation layer


#This is input layer.
def get_pixel_matrix(image_path):
    img = Image.open(image_path)
    width, height = img.size
    pixels = list(img.getdata())
    
    # Reshaping the list of pixels into a 2D matrix
    pixel_matrix = [pixels[i:i+width] for i in range(0, len(pixels), width)]
    
    return pixel_matrix

image_path = 'fl.jpeg'
matrix = get_pixel_matrix(image_path)

#for row in matrix:
#    print(row)
    



#Convolutional layer.
def convolution_layer(matrix, kernel):
    matrix_height, matrix_width = matrix.shape()
    kernel_height, kernel_width = kernel.shape()

    result_height = matrix_height - kernel_height + 1
    result_width = matrix_width - kernel_width + 1

    result = np.zeros((result_height, result_width))

    for i in range(result_height):
        for j in range(result_width):
            result[i, j] = np.sum(matrix[i:i+kernel_height, j:j+kernel_width] * kernel)

    return result

# Defining a simple kernel
kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# Applying convolution layer
convolution_result = convolution_layer(matrix, kernel)

for row in convolution_result:
    print(row)





# Activation layer
# Used RELU(x) = max(0,x) function

def relu_activation(matrix):
    return np.maximum(0, matrix)

convolution_result_matrix = convolution_result

# Apply ReLU activation
activation_result = relu_activation(convolution_result_matrix)

for row in activation_result:
    print(row)
    
    