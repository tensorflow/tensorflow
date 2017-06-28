# =============================================================================
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy
import tensorflow
from tensorflow.contrib.periodic_resample import periodic_resample


class PeriodicResampleTest(tensorflow.test.TestCase):
    def testPeriodicResample(self):

        # basic 2-D tensor
        input_tensor = numpy.arange(12).reshape((3, 4))
        desired_shape = numpy.array([6, -1])
        output_tensor = input_tensor.reshape((6, 2))
        with self.test_session():
            result = periodic_resample(input_tensor, desired_shape).eval()
            self.assertAllEqual(result, output_tensor)

        # basic 2-D tensor (truncated)
        input_tensor = numpy.arange(12).reshape((3, 4))
        desired_shape = numpy.array([5, -1])
        output_tensor = input_tensor.reshape((6, 2))[:-1]
        with self.test_session():
            result = periodic_resample(input_tensor, desired_shape).eval()
            self.assertAllEqual(result, output_tensor)

        # basic 3-D tensor
        input_tensor = numpy.arange(2*2*4).reshape((2, 2, 4))
        desired_shape = numpy.array([4, 4, -1])
        output_tensor = numpy.array([[[0], [2], [4], [6]],
                                     [[1], [3], [5], [7]],
                                     [[8], [10],[12],[14]],
                                     [[9], [11],[13],[15]]])
        # NOTE: output_tensor != input_tensor.reshape((4, 4, -1))
        with self.test_session():
            result = periodic_resample(input_tensor, desired_shape).eval()
            # input_tensor[0, 0, 0] == result[0, 0, 0]
            # input_tensor[0, 0, 1] == result[1, 0, 0]
            # input_tensor[0, 0, 2] == result[0, 1, 0]
            # input_tensor[0, 0, 3] == result[1, 1, 0]
            self.assertAllEqual(result, output_tensor)

        # basic 4-D tensor
        input_tensor = numpy.arange(2*2*2*8).reshape((2, 2, 2, 8))
        desired_shape = numpy.array([4, 4, 4, -1])
        output_tensor = numpy.array([[[[0], [4], [8], [12]],
                                      [[2], [6], [10],[14]],
                                      [[16],[20],[24],[28]],
                                      [[18],[22],[26],[30]]],
                                     [[[1], [5], [9], [13]],
                                      [[3], [7], [11],[15]],
                                      [[17],[21],[25],[29]],
                                      [[19],[23],[27],[31]]],
                                     [[[32],[36],[40],[44]],
                                      [[34],[38],[42],[46]],
                                      [[48],[52],[56],[60]],
                                      [[50],[54],[58],[62]]],
                                     [[[33],[37],[41],[45]],
                                      [[35],[39],[43],[47]],
                                      [[49],[53],[57],[61]],
                                      [[51],[55],[59],[63]]]])
        # NOTE: output_tensor != input_tensor.reshape((4, 4, 4, -1))
        with self.test_session():
            result = periodic_resample(input_tensor, desired_shape).eval()
            self.assertAllEqual(result, output_tensor)


if __name__ == "__main__":
    tensorflow.test.main()
