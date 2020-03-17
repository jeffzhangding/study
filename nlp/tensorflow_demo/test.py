__author__ = 'jeff'

# import seaborn as sns
#
# import matplotlib.pyplot as plt
# sns.set(style="ticks")
#
# # Load the example dataset for Anscombe's quartet
# df = sns.load_dataset("anscombe")
#
# # Show the results of a linear regression within each dataset
# sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
#            col_wrap=2, ci=None, palette="muted", height=4,
#            scatter_kws={"s": 50, "alpha": 1})
#
# plt.show()
# print('==')

import tensorflow as tf

# elements = [[1, 2],
#              [3, 4, 5],
#              [6, 7],
#              [8]]
# A = tf.data.Dataset.from_generator(lambda: iter(elements), tf.int32)
# # Pad to the smallest per-batch size that fits all elements.
# B = A.padded_batch(3, padded_shapes=[None])
# for element in B.as_numpy_iterator():
#     print(element)
# # [[1 2 0]
# # [3 4 5]]
# # [[6 7]
# # [8 0]]
# # Pad to a fixed size.
# C = A.padded_batch(2, padded_shapes=3)
# for element in C.as_numpy_iterator():
#     print(element)
# # [[1 2 0]
# # [3 4 5]]
# # [[6 7 0]
# # [8 0 0]]
# # Pad with a custom value.
# D = A.padded_batch(2, padded_shapes=3, padding_values=-1)
# for element in D.as_numpy_iterator():
#     print(element)
# # [[ 1  2 -1]
# # [ 3  4  5]]
# # [[ 6  7 -1]
# # [ 8 -1 -1]]
# # Components of nested elements can be padded independently.
# elements = [([1, 2, 3], [10]),
#          ([4, 5], [11, 12])]
# dataset = tf.data.Dataset.from_generator(
#  lambda: iter(elements), (tf.int32, tf.int32))
# # Pad the first component of the tuple to length 4, and the second
# # component to the smallest size that fits.
# dataset = dataset.padded_batch(2,
#  padded_shapes=([4], [None]),
#  padding_values=(-1, 100))
# print(list(dataset.as_numpy_iterator()))
# [(array([[ 1,  2,  3, -1], [ 4,  5, -1, -1]], dtype=int32),
# array([[ 10, 100], [ 11,  12]], dtype=int32))]


import numpy as np

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)

print(output_array)
