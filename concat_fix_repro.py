import tensorflow as tf

class TestModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(32)
        self.d3 = tf.keras.layers.Dense(16)

    def call(self, x, indices=None):
        x = self.d1(x)
        if indices is not None:
            (unique_vals, _) = tf.unique(indices)
            x = tf.nn.relu(tf.gather(x, unique_vals))
        else:
            x = tf.nn.relu(x)
        partitioned = tf.dynamic_partition(x, tf.cast(tf.reduce_sum(x, axis=1) > 0, tf.int32), num_partitions=2)
        x = tf.concat(partitioned, axis=0)
        (top_k_values, _) = tf.nn.top_k(x, k=tf.shape(x)[0] // 2)
        x = tf.nn.relu(self.d2(top_k_values))
        return self.d3(x)


def get_default_model():
    return TestModel()


def get_sample_inputs():
    x = tf.random.normal([10, 64])
    indices = tf.random.uniform([10], maxval=5, dtype=tf.int32)
    return (x, indices)


def main():
    model = get_default_model()
    inputs = get_sample_inputs()
    eager_out = model(*inputs)
    print('Eager Input shape:', inputs[0].shape)
    print('Eager Output shape:', eager_out.shape)
    @tf.function(jit_compile=True)
    def compiled_forward(*args):
        return model(*args)
    compiled_out = compiled_forward(*inputs)
    print('XLA Output shape:', compiled_out.shape)


if __name__ == '__main__':
    main()
