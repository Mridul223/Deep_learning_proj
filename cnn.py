import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Reshape((28, 28, 1)),

    tf.keras.layers.Conv2D(8, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training started...", flush=True)

# Train model
model.fit(X_train, y_train, epochs=2, batch_size=128, verbose=1)

print("Training completed!", flush=True)

# Test model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy:", accuracy)

# Predict one image
prediction = model.predict(X_test[:1], verbose=0)
print("Predicted digit:", prediction[0].argmax())
print("Actual digit:", y_test[0])

# Show image
plt.imshow(X_test[0], cmap="gray")
plt.title(f"Predicted: {prediction[0].argmax()}")
plt.show()