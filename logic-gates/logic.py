import tensorflow as tf

def create_model(num_layers = 2, num_units = 15):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer([2]))
  for x in range(num_layers):
    model.add(tf.keras.layers.Dense(num_units, activation="relu"))
  model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

  model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['binary_accuracy'])

  return model

def train_model(model, inp, out):
  inputs = tf.constant(inp)
  outputs = tf.constant(out)
  model.fit(inputs, outputs, epochs=50)

def test_model(model, inp):
  for input in inp:
    result = model(tf.constant([input])).numpy()[0][0]
    print(input, "=>", result)
