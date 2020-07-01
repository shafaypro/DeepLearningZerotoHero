from keras import layers  # Layers
from keras import models  # Models
from keras.datasets import mnist  # Mnist Data set
from keras.datasets import mnist

# Loading in the training testing data sets with respect to images
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)

# Transformation of the images shapes

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Converting the output values to categories
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)




# Creating a Two layer network with 512 Neurons and then 10 neurons
# Input layer -> Hidden(Dense Layer) -> Dense Layer(10) -> Output layer
network = models.Sequential()  # Sequential Modelling
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
# Network compilation using Root Mean Squared proporation
# Cross Entropy (back pass loss correction)
# metrics to use for Accuracy !
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# fitting the model
# Specify the hyper parameters such as batch sizing and Epoch !
network.fit(train_images, train_labels, batch_size=128, epochs=5)
