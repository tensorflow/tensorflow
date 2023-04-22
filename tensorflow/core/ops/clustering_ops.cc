// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
// ==============================================================================

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("KmeansPlusPlusInitialization")
    .Input("points: float32")
    .Input("num_to_sample: int64")
    .Input("seed: int64")
    .Input("num_retries_per_sample: int64")
    .Output("samples: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("KMC2ChainInitialization")
    .Input("distances: float32")
    .Input("seed: int64")
    .Output("index: int64")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("NearestNeighbors")
    .Input("points: float32")
    .Input("centers: float32")
    .Input("k: int64")
    .Output("nearest_center_indices: int64")
    .Output("nearest_center_distances: float32")
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
