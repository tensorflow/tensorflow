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

import random

from sklearn import datasets
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.python.platform import googletest

from skflow.io import *
import skflow


class IOTest(googletest.TestCase):

    def testPandasDataframe(self):
        if HAS_PANDAS:
            random.seed(42)
            iris = datasets.load_iris()
            data = pd.DataFrame(iris.data)
            labels = pd.DataFrame(iris.target)
            classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
            classifier.fit(data, labels)
            score = accuracy_score(classifier.predict(data), labels)
            self.assertGreater(score, 0.5, "Failed with score = {0}".format(score))
        else:
            print("No pandas installed. test_pandas_dataframe skipped.")

    def testPandasSeries(self):
        if HAS_PANDAS:
            random.seed(42)
            iris = datasets.load_iris()
            data = pd.DataFrame(iris.data)
            labels = pd.Series(iris.target)
            classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
            classifier.fit(data, labels)
            score = accuracy_score(classifier.predict(data), labels)
            self.assertGreater(score, 0.5, "Failed with score = {0}".format(score))
        else:
            print("No pandas installed. test_pandas_series skipped.")

    def testStringDataFormats(self):
        with self.assertRaises(ValueError):
            skflow.io.extract_pandas_data(pd.DataFrame({"Test": ["A", "B"]}))
        with self.assertRaises(ValueError):
            skflow.io.extract_pandas_labels(pd.DataFrame({"Test": ["A", "B"]}))
   
    def testDaskIO(self):
        df = pd.DataFrame(dict(a=list('aabbcc'), b=list(range(6))),
                          index=pd.date_range(start='20100101', periods=6))
        ddf=dd.from_pandas(df, npartitions=3)
        extracted_ddf=extract_dask_data(ddf)
        self.assertEqual(extracted_ddf.divisions, (0,2,4,6),
                         "Failed with divisions = {0}".format(extracted_ddf.divisions))

if __name__ == '__main__':
    tf.test.main()
