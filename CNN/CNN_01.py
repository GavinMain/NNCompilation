#Version: CNN_01 optimizes convolution by converting patches to vectors for matrix multiplication
import numpy as np
import time
import torch
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#Hyper Parameters
num_epochs = 1
lr = .001
batch_size = 64

#Other Parameters
log_file = "log.txt"

def relu(x):
    return np.maximum(0, x)

def relu_gradient(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    result = exp_x / sum_exp_x
    return result

def cross_entropy_loss(predictions, targets, eps=1e-12):
    predictions = np.clip(predictions, eps, 1.0)
    loss = -np.sum(targets * np.log(predictions), axis=1)
    return np.mean(loss)

def cross_entropy_loss_gradient(predictions, targets):
    gradient = predictions - targets
    return gradient

class LinearLayer:
  def __init__(self, input_size, output_size):
      self.weights =np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
      self.bias = np.zeros((1, output_size))

  def forward(self, input_data):
      self.input = input_data

      self.out = np.dot(self.input, self.weights) + self.bias
      return self.out

  def backward(self, gradient, learning_rate):
      weights_gradient = np.dot(self.input.T, gradient) / self.input.shape[0]
      bias_gradient = np.sum(gradient, axis=0, keepdims=True) / self.input.shape[0]

      self.weights -= learning_rate * weights_gradient
      self.bias -= learning_rate * bias_gradient

      return np.dot(gradient, self.weights.T)

def train_model(model, train_data, train_labels, learning_rate, batch_size):
  for i in range(0, len(train_data), batch_size):
    batch_data = train_data[i:i+batch_size]
    batch_labels = train_labels[i:i+batch_size]
    out = model.forward(batch_data)
    predictions = softmax(out)
    loss = cross_entropy_loss(predictions, batch_labels)
    gradient = cross_entropy_loss_gradient(predictions, batch_labels)
    model.backward(gradient, learning_rate)
  return np.average(loss)

def evaluate_model(model, test_data, test_labels):
  out = model.forward(test_data)
  out = softmax(out)
  predictions = np.argmax(out, axis=1)
  true_labels = np.argmax(test_labels, axis=1)
  correct_predictions = np.sum(predictions == true_labels)
  return correct_predictions / len(test_labels)

class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))  

    def forward(self, input_data):
        self.input = input_data
        self.mean = np.mean(input_data, axis=1, keepdims=True)
        self.var = np.var(input_data, axis=1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)

        self.norm = (input_data - self.mean) / self.std
        out = self.gamma * self.norm + self.beta
        return out

    def backward(self, grad_output, learning_rate=1e-3):
        _, num_features = grad_output.shape

        gamma_gradient = np.sum(grad_output * self.norm, axis=0, keepdims=True)
        beta_gradient = np.sum(grad_output, axis=0, keepdims=True)

        norm_gradient = grad_output * self.gamma
        var_gradient = np.sum(norm_gradient * (self.input - self.mean) * -0.5 * (self.var + self.eps)**-1.5, axis=1, keepdims=True)
        mean_gradient = np.sum(norm_gradient * -1 / self.std, axis=1, keepdims=True) + var_gradient * np.mean(-2 * (self.input - self.mean), axis=1, keepdims=True)

        return_gradient = norm_gradient / self.std + var_gradient * 2 * (self.input - self.mean) / num_features + mean_gradient / num_features

        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient

        return return_gradient
    
def patch_to_vector(input_data, kernel_height, kernel_width, stride, padding):
    batch_size, in_channels, height, width = input_data.shape

    output_height = (height + 2 * padding - kernel_height) // stride + 1
    output_width = (width + 2 * padding - kernel_width) // stride + 1

    if padding > 0:
        input_padded = np.pad(
            input_data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        input_padded = input_data

    cols = np.zeros((in_channels * kernel_height * kernel_width, output_height * output_width * batch_size), dtype=input_data.dtype)

    col_idx = 0
    for h_out in range(output_height):
        for w_out in range(output_width):
            h_start = h_out * stride
            h_end = h_start + kernel_height
            w_start = w_out * stride
            w_end = w_start + kernel_width

            patch = input_padded[:, :, h_start:h_end, w_start:w_end]
            cols[:, col_idx * batch_size : (col_idx + 1) * batch_size] = patch.reshape(batch_size, -1).T
            col_idx += 1

    return cols

def vector_to_patch(cols, input_shape, kernel_height, kernel_width, stride, padding):
    batch_size, in_channels, height, width = input_shape

    output_height = (height + 2 * padding - kernel_height) // stride + 1
    output_width = (width + 2 * padding - kernel_width) // stride + 1

    if padding > 0:
        d_input_padded = np.zeros((
            batch_size, in_channels, height + 2 * padding, width + 2 * padding
        ), dtype=cols.dtype)
    else:
        d_input_padded = np.zeros(input_shape, dtype=cols.dtype)

    patches = cols.T.reshape(batch_size, -1, output_height * output_width)

    col_idx = 0
    for h_out in range(output_height):
        for w_out in range(output_width):
            h_start = h_out * stride
            h_end = h_start + kernel_height
            w_start = w_out * stride
            w_end = w_start + kernel_width

            patch_grad = patches[:, :, col_idx].reshape(
                batch_size, in_channels, kernel_height, kernel_width
            )
            d_input_padded[:, :, h_start:h_end, w_start:w_end] += patch_grad
            col_idx += 1

    if padding > 0:
        d_input = d_input_padded[:, :, padding:-padding, padding:-padding]
    else:
        d_input = d_input_padded

    return d_input

class ConvolutionLayer2D:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.weights = np.random.uniform(-1, 1, (out_channels, in_channels, kernel_size, kernel_size))
    self.bias = np.zeros(out_channels)

  def forward(self, input):
    self.input = input
    self.input_shape = input.shape
    batch_size, in_channels, height, width = input.shape

    self.output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
    self.output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

    cols = patch_to_vector(input, self.kernel_size, self.kernel_size, self.stride, self.padding)
    weights_reshaped = self.weights.reshape(self.out_channels, -1)

    output_matrix = np.dot(weights_reshaped, cols)
    
    output = output_matrix + self.bias.reshape(-1, 1)
    output = output.reshape(self.out_channels, self.output_height, self.output_width, batch_size)
    output = output.transpose(3, 0, 1, 2) 
    
    self.cols = cols
    self.weights_reshaped = weights_reshaped
    self.out = output
    return output

  def backward(self, gradient, learning_rate):
    batch_size = self.input_shape[0]

    grad_reshaped = gradient.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

    d_weights_reshaped = np.dot(grad_reshaped, self.cols.T) / batch_size
    d_bias = np.sum(grad_reshaped, axis=1) / batch_size

    d_weights = d_weights_reshaped.reshape(self.weights.shape)

    self.weights -= learning_rate * d_weights
    self.bias -= learning_rate * d_bias

    d_cols = np.dot(self.weights_reshaped.T, grad_reshaped)

    d_input = vector_to_patch(d_cols, self.input_shape, self.kernel_size, self.kernel_size, self.stride, self.padding)

    return d_input

class CNN:
  def __init__(self):
      self.conv1 = ConvolutionLayer2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
      self.conv2 = ConvolutionLayer2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
      self.linear = LinearLayer(32 * 28 * 28, 10)

  def forward(self, input_data):
      input_data = self.conv1.forward(input_data) 
      input_data = relu(input_data)
      input_data = self.conv2.forward(input_data) 
      input_data = relu(input_data)
      
      self.flatten_shape = input_data.shape  
      input_data = input_data.reshape(input_data.shape[0], -1)
      input_data = self.linear.forward(input_data)
      return input_data

  def backward(self, gradient, learning_rate):
      gradient = self.linear.backward(gradient, learning_rate)  
      gradient = gradient.reshape(self.flatten_shape)
      
      gradient = gradient * relu_gradient(self.conv2.out)
      gradient = self.conv2.backward(gradient, learning_rate)
      gradient = gradient * relu_gradient(self.conv1.out)
      gradient = self.conv1.backward(gradient, learning_rate)

      return gradient

if __name__ == "__main__":
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  print("Images Before Processing:")
  print("Train Shape:", train_images.shape)
  print("Test Shape:", test_images.shape)
  print("Type:", train_images.dtype)
  print("Min, Max:", np.min(train_images), np.max(train_images))

  #Converts type to float64 and Scale to max=1. No Flatten
  train_images = train_images.reshape((train_images.shape[0], 1, 28, 28)).astype('float64') / 255
  test_images = test_images.reshape((test_images.shape[0], 1, 28, 28)).astype('float64') / 255

  print("\nImages After Processing:")
  print("Train Shape:", train_images.shape)
  print("Test Shape:", test_images.shape)
  print("Type:", train_images.dtype)
  print("Min, Max:", np.min(train_images), np.max(train_images))

  print("\nLabels Before Processing:")
  print("Train Shape:", train_labels.shape)
  print("Test Shape:", test_labels.shape)
  print("Type:", train_labels.dtype)
  print("One Example:", train_labels[0])

  #Converts single number labels into a probability distribution
  #Results in float64 labels, which is why the images get casted into the same type
  train_labels = to_categorical(train_labels)
  test_labels = to_categorical(test_labels)

  print("\nLabels After Processing:")
  print("Train Shape:", train_labels.shape)
  print("Test Shape:", test_labels.shape)
  print("Type:", train_labels.dtype)
  print("One Example:", train_labels[0])
  
  print("Image example (first image in training set):")
  
  plt.imshow(train_images[0].reshape(28, 28), cmap='gray')
  plt.title(f"Label: {np.argmax(train_labels[0])}")
  plt.show()

  model = CNN()

  with open(log_file, 'a') as f:
      f.write("\nCNN_01:\n")

  for epoch in range(1, num_epochs+1):
      start_time = time.time()
      train_loss = train_model(model, train_images, train_labels, lr, batch_size)
      test_acc = evaluate_model(model, test_images, test_labels)

      print(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Accuracy: {test_acc} | Time: {time.time() - start_time}")

      with open(log_file, 'a') as f:
          f.write(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Accuracy: {test_acc} | Time: {time.time() - start_time}\n")

