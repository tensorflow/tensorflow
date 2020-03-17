import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer


class AdaptiveAveragePooling1D(Layer):
    """Average Pooling with adaptive kernel size and strides.
        Arguments:
            bins: Number of steps in the output
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, steps, features)` while `channels_first`
                corresponds to inputs with shape
                `(batch, features, steps)`.

        Input shape:
            - If `data_format='channels_last'`:
                3D tensor with shape `(batch_size, steps, features)`.
            - If `data_format='channels_first'`:
                3D tensor with shape `(batch_size, features, steps)`.

        Output shape:
            - If `data_format='channels_last'`:
                3D tensor with shape `(batch_size, bins, features)`.
            - If `data_format='channels_first'`:
                3D tensor with shape `(batch_size, features, bins)`.
    """
    def __init__(self, bins, data_format='channels_last', **kwargs):
        self._bins = bins
        self.data_format = data_format
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, *args):
        if self.data_format == 'channels_last':
            splits = tf.split(inputs, self._bins, axis=1)
            splits = tf.stack(splits, axis=1)
            out_vect = tf.reduce_mean(splits, axis=2)
        else:
            splits = tf.split(inputs, self._bins, axis=2)
            splits = tf.stack(splits, axis=2)
            out_vect = tf.reduce_mean(splits, axis=3)
        return out_vect


class AdaptiveAveragePooling2D(Layer):
    """Average Pooling with adaptive kernel size and strides.
        Arguments:
            h_bins: Output Height
            w_bins: Output Width

        Input shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, height, width, features)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, features, height, width)`.

        Output shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, h_bins, w_bins, features)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, features, h_bins, w_bins)`.
    """
    def __init__(self, h_bins, w_bins, data_format='channels_last', **kwargs):
        self._h_bins = h_bins
        self._w_bins = w_bins
        self.data_format = data_format
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, *args):
        if self.data_format == 'channels_last':
            split_cols = tf.split(inputs, self._h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, self._w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            out_vect = tf.reduce_mean(split_rows, axis=[2, 4])
        else:
            split_cols = tf.split(inputs, self._h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, self._w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            out_vect = tf.reduce_mean(split_rows, axis=[3, 5])
        return out_vect


class AdaptiveAveragePooling3D(Layer):
    """Average Pooling with adaptive kernel size and strides.
        Arguments:
            h_bins: Output Height
            w_bins: Output Width
            d_bins: Output Depth

            Input shape:
                - If `data_format='channels_last'`:
                    5D tensor with shape `(batch_size, height, width, depth, features)`.
                - If `data_format='channels_first'`:
                    5D tensor with shape `(batch_size, features, height, width, depth)`.

            Output shape:
                - If `data_format='channels_last'`:
                    5D tensor with shape `(batch_size, h_bins, w_bins, d_bins, features)`.
                - If `data_format='channels_first'`:
                    5D tensor with shape `(batch_size, features, h_bins, w_bins, d_bins)`.
        """
    def __init__(self, h_bins, w_bins, d_bins, data_format='channels_last', **kwargs):
        self._h_bins = h_bins
        self._w_bins = w_bins
        self._d_bins = d_bins
        self.data_format = data_format
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, *args):
        if self.data_format == 'channels_last':
            split_cols = tf.split(inputs, self._h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, self._w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            split_depth = tf.split(split_rows, self._d_bins, axis=5)
            split_depth = tf.stack(split_depth, axis=5)
            out_vect = tf.reduce_mean(split_depth, axis=[2, 4, 6])
        else:
            split_cols = tf.split(inputs, self._h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, self._w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            split_depth = tf.split(split_rows, self._d_bins, axis=6)
            split_depth = tf.stack(split_depth, axis=6)
            out_vect = tf.reduce_mean(split_depth, axis=[3, 5, 7])
        return out_vect


class AdaptiveMaxPooling1D(Layer):
    """Max Pooling with adaptive kernel size and strides.
        Arguments:
            bins: Number of steps in the output
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, steps, features)` while `channels_first`
                corresponds to inputs with shape
                `(batch, features, steps)`.

        Input shape:
            - If `data_format='channels_last'`:
                3D tensor with shape `(batch_size, steps, features)`.
            - If `data_format='channels_first'`:
                3D tensor with shape `(batch_size, features, steps)`.

        Output shape:
            - If `data_format='channels_last'`:
                3D tensor with shape `(batch_size, bins, features)`.
            - If `data_format='channels_first'`:
                3D tensor with shape `(batch_size, features, bins)`.
    """
    def __init__(self, bins, data_format='channels_last', **kwargs):
        self._bins = bins
        self.data_format = data_format
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, *args):
        if self.data_format == 'channels_last':
            splits = tf.split(inputs, self._bins, axis=1)
            splits = tf.stack(splits, axis=1)
            out_vect = tf.reduce_max(splits, axis=2)
        else:
            splits = tf.split(inputs, self._bins, axis=2)
            splits = tf.stack(splits, axis=2)
            out_vect = tf.reduce_max(splits, axis=3)
        return out_vect


class AdaptiveMaxPooling2D(Layer):
    """Max Pooling with adaptive kernel size and strides.
        Arguments:
            h_bins: Output Height
            w_bins: Output Width

        Input shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, height, width, features)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, features, height, width)`.

        Output shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, h_bins, w_bins, features)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, features, h_bins, w_bins)`.
    """
    def __init__(self, h_bins, w_bins, data_format='channels_last', **kwargs):
        self._h_bins = h_bins
        self._w_bins = w_bins
        self.data_format = data_format
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, *args):
        if self.data_format == 'channels_last':
            split_cols = tf.split(inputs, self._h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, self._w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            out_vect = tf.reduce_max(split_rows, axis=[2, 4])
        else:
            split_cols = tf.split(inputs, self._h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, self._w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            out_vect = tf.reduce_max(split_rows, axis=[3, 5])
        return out_vect


class AdaptiveMaxPooling3D(Layer):
    """Max Pooling with adaptive kernel size and strides.
        Arguments:
            h_bins: Output Height
            w_bins: Output Width
            d_bins: Output Depth

            Input shape:
                - If `data_format='channels_last'`:
                    5D tensor with shape `(batch_size, height, width, depth, features)`.
                - If `data_format='channels_first'`:
                    5D tensor with shape `(batch_size, features, height, width, depth)`.

            Output shape:
                - If `data_format='channels_last'`:
                    5D tensor with shape `(batch_size, h_bins, w_bins, d_bins, features)`.
                - If `data_format='channels_first'`:
                    5D tensor with shape `(batch_size, features, h_bins, w_bins, d_bins)`.
        """
    def __init__(self, h_bins, w_bins, d_bins, data_format='channels_last', **kwargs):
        self._h_bins = h_bins
        self._w_bins = w_bins
        self._d_bins = d_bins
        self.data_format = data_format
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, *args):
        if self.data_format == 'channels_last':
            split_cols = tf.split(inputs, self._h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, self._w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            split_depth = tf.split(split_rows, self._d_bins, axis=5)
            split_depth = tf.stack(split_depth, axis=5)
            out_vect = tf.reduce_max(split_depth, axis=[2, 4, 6])
        else:
            split_cols = tf.split(inputs, self._h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, self._w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            split_depth = tf.split(split_rows, self._d_bins, axis=6)
            split_depth = tf.stack(split_depth, axis=6)
            out_vect = tf.reduce_max(split_depth, axis=[3, 5, 7])
        return out_vect

