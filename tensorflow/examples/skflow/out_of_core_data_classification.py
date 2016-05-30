#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets, metrics, cross_validation

import pandas as pd
import dask.dataframe as dd

from tensorflow.contrib import learn

# Sometimes when your dataset is too large to hold in the memory
# you may want to load it into a out-of-core dataframe as provided by dask library
# to firstly draw sample batches and then load into memory for training. 

# Load dataset.
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target,
    test_size=0.2, random_state=42)

# Note that we use iris here just for demo purposes
# You can load your own large dataset into a out-of-core dataframe
# using dask's methods, e.g. read_csv() in dask
# details please see: http://dask.pydata.org/en/latest/dataframe.html

# We firstly load them into pandas dataframe and then convert into dask dataframe
X_train, y_train, X_test, y_test = [pd.DataFrame(data) for data in [X_train, y_train, X_test, y_test]]
X_train, y_train, X_test, y_test = [dd.from_pandas(data, npartitions=2) for data in [X_train, y_train, X_test, y_test]]

# Initialize a TensorFlow linear classifier
classifier = learn.TensorFlowLinearClassifier(n_classes=3)

# Fit the model using training set
classifier.fit(X_train, y_train)
# Make predictions on each partitions of testing data
predictions = X_test.map_partitions(classifier.predict).compute()
# Calculate accuracy
score = metrics.accuracy_score(y_test.compute(), predictions)
