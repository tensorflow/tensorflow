from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes
from tensorflow.python.dlpack.dlpack import from_dlpack, to_dlpack

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

int_dtypes = [
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
    np.uint64
]
float_dtypes = [np.float16, np.float32, np.float64]
complex_dtypes = [np.complex64, np.complex128]
dlpack_dtypes = int_dtypes + float_dtypes + [dtypes.bfloat16]
standard_dtypes = int_dtypes + float_dtypes + complex_dtypes + [np.bool_]


testcase_shapes = [
    (),
    (1,),
    (2, 3),
    (2, 0),
    (0, 7),
    (4, 1, 2)
]


def FormatShapeAndDtype(shape, dtype):
    return "_{}[{}]".format(str(dtype), ",".join(map(str, shape)))


class DLPackTest(parameterized.TestCase, test.TestCase):

    @parameterized.named_parameters({
        "testcase_name": FormatShapeAndDtype(shape, dtype),
        "dtype": dtype,
        "shape": shape} for dtype in dlpack_dtypes for shape in testcase_shapes)
    def testRoundTrip(self, dtype, shape):
        np.random.seed(42)
        np_array = np.random.randint(0, 10, shape)
        tf_tensor = constant_op.constant(np_array, dtype=dtype)
        dlcapsule = to_dlpack(tf_tensor)
        del tf_tensor  # should still work
        tf_tensor2 = from_dlpack(dlcapsule)
        self.assertAllClose(np_array, tf_tensor2)

    def testTensorsCanBeConsumedOnceOnly(self):
        np.random.seed(42)
        np_array = np.random.randint(0, 10, (2, 3, 4))
        tf_tensor = constant_op.constant(np_array, dtype=np.float32)
        dlcapsule = to_dlpack(tf_tensor)
        del tf_tensor  # should still work
        tf_tensor2 = from_dlpack(dlcapsule)
        
        def ConsumeDLPackTensor():
            from_dlpack(dlcapsule)  # Should can be consumed only once
        self.assertRaisesRegex(Exception,
                               ".*a DLPack tensor may be consumed at most once.*",
                               ConsumeDLPackTensor)


if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()
