import numpy as np
import logic

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([[0], [1], [1], [1]])

model = logic.create_model()
logic.train_model(model, np.tile(inputs, (20, 1)), np.tile(outputs, (20, 1)))
logic.test_model(model, inputs)
