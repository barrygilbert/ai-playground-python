import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(15, input_dim=2, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['binary_accuracy'])

inputs = np.array([
  [0,0], [0,1], [1,0], [1,1], [0,0], [0,1], [1,0], [1,1], [0,0], [0,1], [1,0], [1,1]
], "float32")

outputs = np.array([ [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0] ], "float32")

model.fit(inputs, outputs, batch_size=1, epochs=50)

test_inputs = [[0,0], [0,1], [1,0], [1,1]]
for input in test_inputs:
  result = model(np.array([input], "float32")).numpy()[0][0]
  print(input, "=>", result)
