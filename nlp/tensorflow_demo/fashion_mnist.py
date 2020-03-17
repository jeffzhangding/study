__author__ = 'jeff'

import os
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

fashion_mnist_data = keras.datasets.fashion_mnist

(tran_images, train_lables), (test_images, test_labels) = fashion_mnist_data.load_data()

# plt.figure()
# plt.imshow(tran_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


tran_images = tran_images / 255.0
test_images = test_images / 255.0


# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(tran_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_lables[i]])
#
# plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(
    optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
              )

model.fit(tran_images, train_lables, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)


print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])



