import tensorflow as tf

optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

def loss(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real,0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

def accuracy(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real,0))
  mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)

  pred *=mask
  acc = train_accuracy(real, pred)
  return tf.reduce_mean(acc)
  

