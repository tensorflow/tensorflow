import os
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn_ops

cur_dir = os.path.dirname(os.path.realpath(__file__))


def shape_nchw_to_nhwc(shape):
    return (shape[0], shape[2], shape[3], shape[1])


class TestBnns(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        so_path = (cur_dir + "/../../../" +
            "bazel-bin/tensorflow/core/user_ops/bnns.so")

        assert os.path.isfile(so_path), \
            "Please build bnns operation with\n" \
            "./configure && " \
            "bazel build --config opt //tensorflow/core/user_ops:bnns.so\n" \
            "from tensorflow root"
        cls.bnns_module = tf.load_op_library(so_path)


class TestConv2dBnns(TestBnns):
    def test_valid_padding(self):
        strides = [1,1,2,1]
        padding = 'VALID'
        self._run_test(padding, strides)

    def test_valid_padding_fp16(self):
        strides = [1,1,2,1]
        padding = 'VALID'
        mult = 1. / 10000.
        fp16_0 = self._run_test(
            padding, strides,
            weights_data_type='float16',
            input_data_type='float16',
            rtol=1e-3,
            mult=mult
        )
        fp16_1 = self._run_test(
            padding, strides,
            weights_data_type='float16',
            mult=mult,
            rtol=1e-3,
        )
        fp16_2 = self._run_test(
            padding, strides,
            input_data_type='float16',
            mult=mult,
            rtol=1e-4,
        )
        fp32 = self._run_test(
            padding, strides,
            mult=mult,
            rtol=1e-4,
        )
        self.assert_(not (fp32 == fp16_0).all())
        self.assert_(not (fp32 == fp16_1).all())
        self.assert_(not (fp32 == fp16_2).all())
        self.assert_(not (fp16_0 == fp16_1).all())
        self.assert_(not (fp16_0 == fp16_2).all())
        self.assert_(not (fp16_1 == fp16_2).all())

    def test_same_padding(self):
        strides = [1,1,1,1]
        padding = 'SAME'
        self._run_test(padding, strides)

    def test_linear_activation(self):
        strides = [1,1,2,1]
        padding = 'VALID'
        self._run_test(padding, strides, activation_function='Linear')

    def _run_test(self, padding, strides, mult=None, rtol=1e-7, **kwargs):
        images = (np.arange(180.)-90.).reshape((3, 5, 3, 4)).astype(np.float32)
        images_t = images.transpose(0, 2, 3, 1)
        if mult is not None:
            images_t *= mult

        filters = np.zeros((3, 3, 5, 2), np.float32)
        filters[:, :, 0, 0] = 1
        filters[:, :, 0, 1] = 2
        if mult is not None:
            filters *= mult

        with self.test_session():
            # TODO: Figure out if there is someway for bias to be optional:
            # http://stackoverflow.com/questions/42754965/marking-input-as-optional-in-tensorflow
            bnns_output = self.bnns_module.conv2dbnns(
                images,
                filters,
                strides=strides,
                padding=padding,
                data_format='NCHW',
                **kwargs
            ).eval()
        with self.test_session():
            gold_output = nn_ops.conv2d(
                images_t,
                filters,
                strides=shape_nchw_to_nhwc(strides),
                padding=padding,
            ).eval()
        np.testing.assert_allclose(
            bnns_output,
            gold_output.transpose(0, 3, 1, 2),
            rtol=rtol,
        )
        return bnns_output


class TestConv2dBnnsBias(TestBnns):
    def test_valid_padding(self):
        strides = [1,1,1,1]
        padding = 'VALID'
        self._run_test(padding, strides)

    def _run_test(self, padding, strides):
        images = (np.arange(180.)-90.).reshape((3, 5, 3, 4)).astype(np.float32)
        images_t = images.transpose(0, 2, 3, 1)
        filters = np.zeros((3, 3, 5, 2), np.float32)
        filters[:, :, 0, 0] = 1
        filters[:, :, 0, 1] = 2

        bias = np.arange(1., 3.)

        with self.test_session():
            bnns_output = self.bnns_module.conv2dbnns_with_bias(
                images,
                filters,
                bias=bias,
                strides=strides,
                padding=padding,
                data_format='NCHW',
            ).eval()
        with self.test_session():
            value = nn_ops.conv2d(
                images_t,
                filters,
                strides=shape_nchw_to_nhwc(strides),
                padding=padding,
            )
            gold_output = nn_ops.bias_add(
                value,
                bias,
            ).eval()
        np.testing.assert_allclose(
            bnns_output,
            gold_output.transpose(0, 3, 1, 2),
        )


class TestConv2dBnnsReLU(TestBnns):
    def test_valid_padding(self):
        strides = [1,1,1,1]
        padding = 'VALID'
        self._run_test(padding, strides)

    def _run_test(self, padding, strides):
        images = (np.arange(180.)-90.).reshape((3, 5, 3, 4)).astype(np.float32)
        images_t = images.transpose(0, 2, 3, 1)
        filters = np.zeros((3, 3, 5, 2), np.float32)
        filters[:, :, 0, 0] = 1
        filters[:, :, 0, 1] = 2

        with self.test_session():
            bnns_output = self.bnns_module.conv2dbnns(
                images,
                filters,
                strides=strides,
                padding=padding,
                activation_function='ReLU',
                data_format='NCHW',
            ).eval()
        with self.test_session():
            value = nn_ops.conv2d(
                images_t,
                filters,
                strides=shape_nchw_to_nhwc(strides),
                padding=padding,
            )
            gold_output = nn_ops.relu(value).eval()
        np.testing.assert_allclose(
            bnns_output,
            gold_output.transpose(0, 3, 1, 2),
        )

class TestMaxPoolBnns(TestBnns):
    def test_valid_padding(self):
        ksize = [1,1,3,3]
        strides = [1,1,2,2]
        padding = 'VALID'

        self._run_test(ksize, padding, strides)

    def test_same_padding(self):
        ksize = [1,1,3,3]
        strides = [1,1,2,2]
        padding = 'SAME'

        self._run_test(ksize, padding, strides)

    def _run_test(self, ksize, padding, strides):
        images = np.arange(720.).reshape((3, 5, 6, 8)).astype(np.float32)
        images_t = images.transpose(0, 2, 3, 1)
        with self.test_session():
            bnns_output = self.bnns_module.max_pool_bnns(
                images,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format='NCHW',
            ).eval()
        with self.test_session():
            gold_output = nn_ops.max_pool(
                images_t,
                ksize=shape_nchw_to_nhwc(ksize),
                strides=shape_nchw_to_nhwc(strides),
                padding=padding,
            ).eval()
        np.testing.assert_allclose(
            bnns_output,
            gold_output.transpose(0, 3, 1, 2),
        )


if __name__ == "__main__":
    tf.test.main()
