import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),  # 4 input features
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

num_neurons_to_replace = 2

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)

baseline_score = model.evaluate(X_test, y_test)

# model.save('baseline_model.h5')

baseline_model = tf.keras.models.load_model('baseline_model.h5')

layer_to_modify = baseline_model.layers[0]

neurons_to_replace = np.random.choice(layer_to_modify.units, size=num_neurons_to_replace, replace=False)

weights, biases = layer_to_modify.get_weights()
new_weights = np.random.randn(weights.shape[0], num_neurons_to_replace)  # Generate new random weights
weights[:, neurons_to_replace] = new_weights
layer_to_modify.set_weights([weights, biases])

baseline_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
baseline_model.fit(X_train, y_train, epochs=10)
new_score = baseline_model.evaluate(X_test, y_test)

baseline_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

baseline_model.fit(X_train, y_train, epochs=5)

new_score = baseline_model.evaluate(X_test, y_test)
