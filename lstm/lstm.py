import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

inputs = tf.constant([[[1, 2, 3, 4],
                       [4, 6, 7, 8],
                       [9, 10, 11, 12]],
                      [[13, 14, 15, 16],
                       [17, 18, 19, 20],
                       [21, 22, 23, 24]]])

inputs = keras.Input((1, 4), batch_size=2)
lstm_l = layers.LSTM(10, return_sequences=True, return_state=True)
out = lstm_l(inputs)
print(out)
# in_split = tf.split(inputs, 4, axis=1)
#
# cells=[
# layers.LSTMCell(5),
# layers.LSTMCell(5),
# layers.LSTMCell(5),
# layers.LSTMCell(5)
# ]
# lstm_cell = layers.StackedRNNCells(cells, name='stack')(inputs)


model = keras.models.Model(inputs=inputs, outputs=out)
model.summary(line_length=150)
