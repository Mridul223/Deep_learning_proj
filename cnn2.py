# ============================================================
# CNN IMAGE CLASSIFICATION USING MNIST DATASET
# ============================================================

# In this practical, we will:
#
# 1. Import the required libraries
# 2. Load the MNIST handwritten digit dataset
# 3. Understand the shape of the dataset
# 4. Display an image
# 5. Normalize pixel values
# 6. Add the channel dimension
# 7. Create a CNN model
# 8. Add convolution layers
# 9. Add pooling layers
# 10. Flatten the feature maps
# 11. Add Dense layers
# 12. Compile the model
# 13. Train the model
# 14. Evaluate the model
# 15. Make predictions
#
# Our goal:
# Given an image of a handwritten digit,
# the CNN should predict whether it is:
#
# 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9


# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================


# TensorFlow is the main deep learning library
# that we will use to build and train our CNN.
import tensorflow as tf


# Matplotlib is used for visualization.
# We will use it to display an MNIST image.
import matplotlib.pyplot as plt


# Sequential allows us to create a neural network
# by adding layers one after another.
#
# Example:
#
# Input
#   ↓
# Conv2D
#   ↓
# Pooling
#   ↓
# Dense
#   ↓
# Output
#
# This type of architecture is called Sequential.
from tensorflow.keras.models import Sequential


# Import the layers that we need for our CNN.
from tensorflow.keras.layers import (
    Conv2D,          # Performs convolution on the image
    MaxPooling2D,    # Reduces the size of feature maps
    Flatten,         # Converts 3D feature maps into 1D
    Dense            # Fully connected neural network layer
)


# ============================================================
# STEP 2: LOAD THE MNIST DATASET
# ============================================================


# MNIST is a famous dataset used for handwritten digit
# classification.
#
# It contains images of handwritten digits from:
#
# 0
# 1
# 2
# ...
# 9
#
# There are:
#
# 60,000 training images
# 10,000 testing images
#
# Each image is:
#
# 28 pixels × 28 pixels
#
# load_data() returns four things:
#
# X_train -> training images
# y_train -> training labels
# X_test  -> testing images
# y_test  -> testing labels


(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)


# ============================================================
# STEP 3: UNDERSTAND THE DATASET SHAPE
# ============================================================


# Print the shape of the training images.
#
# Expected output:
#
# (60000, 28, 28)
#
# Meaning:
#
# 60000 = number of training images
# 28    = image height
# 28    = image width
#
# At this point there is NO channel dimension yet.
print("Training images:", X_train.shape)


# Print the shape of the training labels.
#
# Expected:
#
# (60000,)
#
# This means we have one label for each training image.
print("Training labels:", y_train.shape)


# Print the shape of the testing images.
#
# Expected:
#
# (10000, 28, 28)
#
# 10000 = number of test images
# 28    = height
# 28    = width
print("Testing images:", X_test.shape)


# Print the shape of the testing labels.
#
# Expected:
#
# (10000,)
#
# One label exists for each testing image.
print("Testing labels:", y_test.shape)


# ============================================================
# STEP 4: DISPLAY ONE IMAGE
# ============================================================


# X_train[0] means:
#
# "Give me the first image from the training dataset."
#
# Python indexing starts from 0.
#
# Therefore:
#
# X_train[0] -> first image
# X_train[1] -> second image
# X_train[2] -> third image
#
# etc.
#
# We use imshow() to display the image.
plt.imshow(
    X_train[1],
    cmap="gray"
)


# Add a title above the image.
#
# y_train[0] contains the actual digit/label
# corresponding to the first image.
#
# For example, if y_train[0] = 5,
# the title will be:
#
# Label: 5
plt.title(
    f"Label: {y_train[0]}"
)


# Display the image on the screen.
plt.show()


# ============================================================
# STEP 5: NORMALIZE PIXEL VALUES
# ============================================================


# Every MNIST image contains pixel values between:
#
# 0 and 255
#
# 0   -> black
# 255 -> white
#
# Neural networks generally work better when input values
# are kept in a smaller range.
#
# Therefore, we divide every pixel value by 255.
#
# Example:
#
# 0   / 255 = 0
# 128 / 255 = 0.50
# 255 / 255 = 1
#
# So after normalization:
#
# Old range:
# 0 to 255
#
# New range:
# 0 to 1


X_train = X_train / 255.0


# Normalize the testing images as well.
#
# IMPORTANT:
# We must apply the same preprocessing to
# training and testing data.
X_test = X_test / 255.0


# ============================================================
# STEP 6: ADD CHANNEL DIMENSION
# ============================================================


# Before this step, our image shape is:
#
# (28, 28)
#
# But CNN expects image data in this format:
#
# (height, width, channels)
#
# For a grayscale image:
#
# height  = 28
# width   = 28
# channels = 1
#
# Therefore:
#
# (28, 28)
#
# becomes:
#
# (28, 28, 1)
#
#
# Why is channel = 1?
#
# Because MNIST images are grayscale.
#
# A grayscale image has only one channel.
#
# An RGB image would have:
#
# (height, width, 3)
#
# because RGB contains:
#
# Red
# Green
# Blue


X_train = X_train.reshape(
    -1,      # Automatically calculate number of images
    28,      # Image height
    28,      # Image width
    1        # Number of channels
)


# Do exactly the same transformation
# for the testing images.
X_test = X_test.reshape(
    -1,
    28,
    28,
    1
)


# After this transformation:
#
# X_train shape becomes:
#
# (60000, 28, 28, 1)
#
# X_test shape becomes:
#
# (10000, 28, 28, 1)


# ============================================================
# STEP 7: CREATE THE CNN MODEL
# ============================================================


# Create an empty Sequential model.
#
# We will now add CNN layers one by one.
#
# The architecture will look like:
#
# Image
#   ↓
# Conv2D
#   ↓
# MaxPooling
#   ↓
# Conv2D
#   ↓
# MaxPooling
#   ↓
# Flatten
#   ↓
# Dense
#   ↓
# Output


model = Sequential()


# ============================================================
# STEP 8: FIRST CONVOLUTION LAYER
# ============================================================


# Add the first convolution layer.
#
# Conv2D is used because our input is a 2D image.
#
# filters=32
# ----------------
# We are asking the CNN to learn 32 different filters.
#
# Each filter can learn a different visual pattern.
#
# For example, conceptually, different filters may learn:
#
# - vertical edges
# - horizontal edges
# - corners
# - curves
# - other useful patterns
#
# IMPORTANT:
# We are NOT manually giving these filters.
# The CNN learns their values during training.
#
#
# kernel_size=(3,3)
# -----------------
# Each filter is 3 pixels × 3 pixels.
#
# The filter moves across the image and performs
# convolution operations.
#
#
# activation="relu"
# ------------------
# ReLU means:
#
# ReLU(x) = max(0, x)
#
# Negative values become 0.
# Positive values remain.
#
# ReLU also introduces non-linearity into the network.
#
#
# input_shape=(28,28,1)
# ---------------------
# This tells the first CNN layer:
#
# Height = 28
# Width = 28
# Channel = 1
#
# Therefore our input image has shape:
#
# 28 × 28 × 1


model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    )
)


# ============================================================
# WHAT HAPPENS AFTER FIRST CONVOLUTION?
# ============================================================


# Input:
#
# 28 × 28 × 1
#
# Filter:
#
# 3 × 3
#
# Since we are using the default padding="valid"
# and stride=1:
#
# Output size:
#
# 28 - 3 + 1 = 26
#
# Therefore the output becomes:
#
# 26 × 26 × 32
#
# Why 32?
#
# Because we used 32 filters.
#
# So:
#
# 28 × 28 × 1
#       ↓
# Conv2D(32, 3×3)
#       ↓
# 26 × 26 × 32
#
# The 32 represents 32 feature maps.


# ============================================================
# STEP 9: FIRST MAX POOLING LAYER
# ============================================================


# MaxPooling2D is used to reduce the spatial dimensions
# of the feature maps.
#
# pool_size=(2,2)
# means we look at a 2 × 2 region.
#
# From each 2 × 2 region, we keep only the largest value.
#
# Example:
#
# 1  5
# 3  2
#
# Maximum = 5
#
# Therefore max pooling keeps the strongest activation.


model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# The feature map size approximately becomes:
#
# 26 × 26 × 32
#       ↓
# MaxPooling(2×2)
#       ↓
# 13 × 13 × 32
#
# Notice:
#
# Height decreases:
# 26 → 13
#
# Width decreases:
# 26 → 13
#
# Number of feature maps stays:
# 32 → 32


# ============================================================
# STEP 10: SECOND CONVOLUTION LAYER
# ============================================================


# Now we add another convolution layer.
#
# This layer has 64 filters.
#
# The first convolution layer learns simpler features.
#
# The second convolution layer can combine those
# simpler features to learn more complex patterns.
#
# Conceptually:
#
# First layer:
# edges
# lines
# corners
#
# Second layer:
# curves
# shapes
# combinations of edges
#
#
# filters=64
# means we learn 64 feature maps.
#
# kernel_size=(3,3)
# means each filter is 3 × 3.
#
# activation="relu"
# applies the ReLU activation function.


model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    )
)


# Shape calculation:
#
# Input:
#
# 13 × 13 × 32
#
# Filter:
#
# 3 × 3
#
# Output height:
#
# 13 - 3 + 1 = 11
#
# Therefore:
#
# 13 × 13 × 32
#       ↓
# Conv2D(64, 3×3)
#       ↓
# 11 × 11 × 64


# ============================================================
# STEP 11: SECOND MAX POOLING LAYER
# ============================================================


# Again we reduce the spatial dimensions
# using max pooling.
#
# We use a 2 × 2 pooling window.
#
# The largest value from each region is retained.


model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# Shape becomes approximately:
#
# 11 × 11 × 64
#       ↓
# MaxPooling(2×2)
#       ↓
# 5 × 5 × 64
#
# So our image has now been transformed from:
#
# 28 × 28 × 1
#
# into:
#
# 5 × 5 × 64
#
# The CNN has extracted useful features
# from the original image.


# ============================================================
# STEP 12: FLATTEN
# ============================================================


# At this point we have:
#
# 5 × 5 × 64
#
# This is a 3-dimensional feature representation.
#
# Dense layers work with a 1-dimensional vector.
#
# Therefore we use Flatten().
#
#
# Calculation:
#
# 5 × 5 × 64
#
# = 25 × 64
#
# = 1600
#
# So Flatten converts:
#
# 5 × 5 × 64
#
# into:
#
# 1600
#
#
# Visual idea:
#
# 3D feature maps
#       ↓
#    Flatten
#       ↓
# 1D vector


model.add(
    Flatten()
)


# ============================================================
# STEP 13: FULLY CONNECTED / DENSE LAYER
# ============================================================


# Now we add a Dense layer.
#
# Dense means every neuron in this layer
# is connected to every input value.
#
#
# units=128
# ----------
# We create 128 neurons.
#
#
# activation="relu"
# ------------------
# We again use ReLU to introduce non-linearity.
#
#
# At this stage, the CNN has already extracted
# visual features.
#
# The Dense layer uses those features to perform
# higher-level reasoning for classification.


model.add(
    Dense(
        128,
        activation="relu"
    )
)


# ============================================================
# STEP 14: OUTPUT LAYER
# ============================================================


# MNIST contains 10 possible classes:
#
# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
#
# Therefore we need 10 output neurons.
#
# Each neuron represents one class.
#
# Example:
#
# Neuron 0 → probability of digit 0
# Neuron 1 → probability of digit 1
# Neuron 2 → probability of digit 2
# ...
# Neuron 9 → probability of digit 9
#
#
# Why softmax?
#
# Softmax converts the output values into probabilities.
#
# Example:
#
# 0 → 0.01
# 1 → 0.02
# 2 → 0.90
# 3 → 0.01
# ...
#
# The probabilities add up to approximately 1.
#
# The class with the highest probability
# becomes the prediction.


model.add(
    Dense(
        10,
        activation="softmax"
    )
)


# ============================================================
# COMPLETE CNN ARCHITECTURE
# ============================================================
#
# Now our complete model is:
#
#
# Input
# 28 × 28 × 1
#       ↓
# Conv2D
# 26 × 26 × 32
#       ↓
# MaxPooling
# 13 × 13 × 32
#       ↓
# Conv2D
# 11 × 11 × 64
#       ↓
# MaxPooling
# 5 × 5 × 64
#       ↓
# Flatten
# 1600
#       ↓
# Dense
# 128
#       ↓
# Dense
# 10
#       ↓
# Softmax
#
# Final output:
#
# Probability for digits 0 to 9


# ============================================================
# STEP 15: DISPLAY MODEL ARCHITECTURE
# ============================================================


# model.summary() displays information about
# every layer in our model.
#
# It shows:
#
# - Layer name
# - Output shape
# - Number of parameters
#
# This is very useful for understanding
# how the data changes after every layer.


model.summary()


# ============================================================
# STEP 16: COMPILE THE MODEL
# ============================================================


# Before training, we need to compile the model.
#
# Compile tells TensorFlow:
#
# 1. Which optimizer to use
# 2. Which loss function to use
# 3. Which metrics we want to track


model.compile(

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------
    #
    # Adam is an optimization algorithm.
    #
    # Its job is to update the model's weights
    # during training so that the loss decreases.
    #
    # Simple meaning:
    #
    # Adam helps the model learn.
    #
    optimizer="adam",


    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------
    #
    # Our problem is multi-class classification.
    #
    # We have 10 classes:
    #
    # 0 to 9
    #
    # sparse_categorical_crossentropy is suitable
    # when labels are integer values such as:
    #
    # 0, 1, 2, 3, ..., 9
    #
    # The loss tells us how wrong the model prediction is.
    #
    loss="sparse_categorical_crossentropy",


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    #
    # We want to monitor accuracy during training.
    #
    # Accuracy tells us the percentage of predictions
    # that are correct.
    #
    metrics=["accuracy"]
)


# ============================================================
# STEP 17: TRAIN THE MODEL
# ============================================================


# model.fit() starts the training process.
#
# The model will look at the training images,
# make predictions, calculate the loss,
# and update its weights.
#
#
# X_train
# --------
# Training images.
#
#
# y_train
# --------
# Correct labels for the training images.


history = model.fit(

    X_train,
    y_train,


    # --------------------------------------------------------
    # EPOCHS
    # --------------------------------------------------------
    #
    # One epoch means the model has seen
    # the complete training dataset once.
    #
    # epochs=5 means:
    #
    # The model will go through the training
    # dataset 5 times.
    #
    epochs=5,


    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------
    #
    # Instead of sending all 60,000 images
    # to the model at once,
    # we process them in smaller groups.
    #
    # batch_size=64 means:
    #
    # The model processes 64 images at a time.
    #
    batch_size=64,


    # --------------------------------------------------------
    # VALIDATION SPLIT
    # --------------------------------------------------------
    #
    # validation_split=0.1 means:
    #
    # 10% of the training data will be kept aside
    # for validation.
    #
    # The model does NOT use this data for weight updates.
    #
    # It is used to check how well the model
    # performs on unseen data during training.
    #
    validation_split=0.1
)


# ============================================================
# STEP 18: EVALUATE THE MODEL
# ============================================================


# After training, we test the model on the
# completely separate testing dataset.
#
# This gives us an idea of how well the model
# generalizes to unseen images.


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)


# Print the final testing accuracy.
#
# Example:
#
# Test Accuracy: 0.98
#
# 0.98 means approximately 98% accuracy.


print(
    "Test Accuracy:",
    test_accuracy
)


# ============================================================
# STEP 19: MAKE A PREDICTION
# ============================================================


# We now want the CNN to predict one image.
#
# X_test[0]
# means the first test image.
#
# But we use:
#
# X_test[0:1]
#
# instead of:
#
# X_test[0]
#
# because the model expects the input to contain
# a batch of images.
#
# X_test[0:1] has shape:
#
# (1, 28, 28, 1)
#
# Here:
#
# 1 = one image
# 28 = height
# 28 = width
# 1 = channel


prediction = model.predict(
    X_test[0:1]
)


# ============================================================
# STEP 20: UNDERSTAND THE PREDICTION
# ============================================================


# prediction contains probabilities for all 10 classes.
#
# Example:
#
# [
#   0.001,   # probability of 0
#   0.002,   # probability of 1
#   0.003,   # probability of 2
#   0.950,   # probability of 3
#   ...
# ]
#
# The class with the highest probability
# is our predicted class.


# ============================================================
# STEP 21: GET THE PREDICTED CLASS
# ============================================================


# argmax() returns the INDEX of the
# largest value.
#
# Example:
#
# probabilities:
#
# [0.01, 0.02, 0.90, 0.03, ...]
#
# The highest value is 0.90.
#
# Its index is 2.
#
# Therefore:
#
# predicted_class = 2


predicted_class = prediction.argmax()


# ============================================================
# STEP 22: DISPLAY PREDICTION
# ============================================================


# Print what the CNN predicted.


print(
    "Predicted:",
    predicted_class
)


# Print the actual correct answer
# stored in the dataset.


print(
    "Actual:",
    y_test[0]
)