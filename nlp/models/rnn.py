__author__ = 'jeff'

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0···"


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# tf.data.TextLineDataset()


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

print('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))

assert original_string == sample_string

for index in encoded_string:
    print('{} ----> {}'.format(index, encoder.decode([index])))


BUFFER_SIZE = 10000
# BATCH_SIZE = 64
BATCH_SIZE = 64


train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# train_dataset = train_dataset.batch(BATCH_SIZE)

train_dataset = train_dataset.padded_batch(BATCH_SIZE, ([None], ()))

test_dataset = test_dataset.padded_batch(BATCH_SIZE, ([None], ()))


# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(encoder.vocab_size, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# back_lstm = tf.keras.layers.LSTM(64)
backward_layer = tf.keras.layers.LSTM(64, return_sequences=True, go_backwards=True)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True), backward_layer=backward_layer),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    # lr = K.get_value(model.optimizer.lr)
    # if lr < 0.0001:
    #     return lr
    # if epoch % 1 == 0 and epoch != 0:
    #     lr = lr * 0.1
    #     print("lr changed to {}".format(lr * 0.1))
    if epoch > 1:
        lr = 1e-4
    return 1e-4


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1.0),
              metrics=['accuracy'])


reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

history = model.fit(train_dataset, epochs=12,
                    validation_data=test_dataset,
                    validation_steps=30,
                    callbacks=[reduce_lr])

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sample_pred_text, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
        encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
        predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)


sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)


plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')


