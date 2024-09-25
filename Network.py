import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load the data
data = np.load('pong_data.npy')
labels = np.load('pong_labels.npy')

# Create the model
model = Sequential([
    Flatten(input_shape=(5,)),  # Input: ball_x, ball_y, ball_speed_x, ball_speed_y, paddle_y
    Dense(24, activation='relu'),
    Dense(3, activation='softmax')  # Output: up, down, stay
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
training = data.shape[0] * 0.8
data = data[:int(training)]
labels = labels[:int(training)]

test_data = data[int(training):]
test_labels = labels[int(training):]

model.fit(data, labels, epochs=200)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)

model.summary()

# Save the model
model.save('pong_ai_model2.h5')
