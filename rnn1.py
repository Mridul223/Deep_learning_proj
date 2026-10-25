import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense ,SimpleRNN

X=np.array([
    [[1],[2],[3]],
    [[2],[3],[4]],
    [[3],[4],[5]],
    [[4],[5],[6]],
    [[5],[6],[7]],
    [[6],[7],[8]],
])

y=np.array([4,5,6,7,8,9,])

print("X shape:",X.shape)
print("y shape:",y.shape)

model=Sequential()
model.add(
    SimpleRNN(
        units=10,
        activation="tanh",
        input_shape=(3,1)
    )
)

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

model.fit(
    X,y,
    epochs=2000,
    verbose=1
)

new=np.array([
    [6],[7],[8]
])

prediction=model.predict(new)
print("perdiction next number:",prediction[0][0])