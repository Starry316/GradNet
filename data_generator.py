import tensorflow as tf


def data_generator(path, batch_size):
    files = tf.data.Dataset.list_files(path)

    def parse_function(example_proto):
        features = tf.io.parse_single_example(example_proto,
                                              features={
                                                  'data': tf.io.FixedLenFeature(
                                                      shape=(256, 256, 12),
                                                      dtype=tf.float32),

                                              })
        return features['data']

    def data_iterator(tfrecords):
        dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=12)
        dataset = dataset.map(map_func=parse_function, num_parallel_calls=12)
        dataset = dataset.repeat(-1)
        # dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    train_iterator = data_iterator(files)
    return train_iterator
