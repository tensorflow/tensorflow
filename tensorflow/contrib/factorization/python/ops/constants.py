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
# ==============================================================================

"""Constants for export/import."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# Euclidean distance between vectors U and V is defined as ||U - V||_F which is
# the square root of the sum of the absolute squares of the elements difference.
SQUARED_EUCLIDEAN_DISTANCE = "squared_euclidean"
# Cosine distance between vectors U and V is defined as
# 1 - (U \dot V) / (||U||_F ||V||_F)
COSINE_DISTANCE = "cosine"

RANDOM_INIT = "random"
KMEANS_PLUS_PLUS_INIT = "kmeans_plus_plus"
KMC2_INIT = "kmc2"

# The name of the variable holding the cluster centers. Used by the Estimator.
CLUSTERS_VAR_NAME = "clusters"

SCALAR_INIT = "scalar"
LIST_INIT = "list"
WORKER_INIT = "worker_init"

# Machine epsilon.
MEPS = np.finfo(float).eps
FULL_COVARIANCE = "full"
DIAG_COVARIANCE = "diag"

#Tensorflow Gaussian mixture model clusters
CLUSTERS_WEIGHT = "alphas"
CLUSTERS_VARIABLE = "clusters"
CLUSTERS_COVS_VARIABLE = "clusters_covs"
