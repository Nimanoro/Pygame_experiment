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
model.fit(data, labels, epochs=1000)

# Save the model
model.save('pong_ai_model2.h5')
