#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Example of loading karge data sets into out-of-core dataframe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
# pylint: disable=g-bad-import-order
import dask.dataframe as dd
import pandas as pd
from tensorflow.contrib import learn
# pylint: enable=g-bad-import-order

# Sometimes when your dataset is too large to hold in the memory
# you may want to load it into a out-of-core dataframe as provided by dask
# library to firstly draw sample batches and then load into memory for training.

# Load dataset.
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# Note that we use iris here just for demo purposes
# You can load your own large dataset into a out-of-core dataframe
# using dask's methods, e.g. read_csv() in dask
# details please see: http://dask.pydata.org/en/latest/dataframe.html

# We firstly load them into pandas dataframe and then convert into dask
# dataframe.
x_train, y_train, x_test, y_test = [
    pd.DataFrame(data) for data in [x_train, y_train, x_test, y_test]]
x_train, y_train, x_test, y_test = [
    dd.from_pandas(data, npartitions=2)
    for data in [x_train, y_train, x_test, y_test]]

# Initialize a TensorFlow linear classifier
classifier = learn.LinearClassifier(
    feature_columns=learn.infer_real_valued_columns_from_input(x_train),
    n_classes=3)

# Fit the model using training set.
classifier.fit(x_train, y_train, steps=200)
# Make predictions on each partitions of testing data
predictions = x_test.map_partitions(classifier.predict).compute()
# Calculate accuracy
score = metrics.accuracy_score(y_test.compute(), predictions)
