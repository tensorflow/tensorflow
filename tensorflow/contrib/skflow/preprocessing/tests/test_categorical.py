# encoding: utf-8

#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pandas
import tensorflow as tf

from skflow.preprocessing import categorical


class CategoricalTest(tf.test.TestCase):

    def testSingleCategoricalProcessor(self):
        cat_processor = categorical.CategoricalProcessor(
            min_frequency=1)
        X = cat_processor.fit_transform(
            [["0"], [1], [float('nan')], 
            ["C"], ["C"], [1], ["0"], [np.nan], [3]])
        self.assertAllEqual(list(X), [
            [2], [1], [0], [3], 
            [3], [1], [2], [0],
            [0]])

    def testSingleCategoricalProcessorPandasSingleDF(self):
        cat_processor = categorical.CategoricalProcessor()
        data = pandas.DataFrame({"Gender": ["Male", "Female", "Male"]})
        X = list(cat_processor.fit_transform(data))
        self.assertAllEqual(list(X), [[1], [2], [1]]) 

    def testMultiCategoricalProcessor(self):
        cat_processor = categorical.CategoricalProcessor(
            min_frequency=0, share=False)
        x = cat_processor.fit_transform(
            [["0", "Male"], [1, "Female"], ["3", "Male"]])
        self.assertAllEqual(list(x), [[1, 1], [2, 2], [3, 1]])


if __name__ == "__main__":
    tf.test.main()
