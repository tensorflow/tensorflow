# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
import unittest

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


class SeededKernelInitializers(unittest.TestCase):

    def test_seeded_initializers(self):
        """
            This test checks that initializer that have seed parameter can use seed provided through Dense layer.
            Initializers like Zeros which do not require seed are not broked by introduced seed.
        """
        seed = 1234

        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu', kernel_initializer='orthogonal', seed=seed))

        config = model.get_config()
        initializer = config['layers'][0]['config']['kernel_initializer']
        self.assertTrue(initializer['class_name'] == 'Orthogonal')
        # seed is bumped up within Orthogonal after being used for initialising random generator
        self.assertEqual(initializer['config']['seed'], seed + 1)


if __name__ == '__main__':
    unittest.main()