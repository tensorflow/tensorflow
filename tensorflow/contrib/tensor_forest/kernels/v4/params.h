// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_PARAMS_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_PARAMS_H_

#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensorforest {

// Return the value of the given depth-dependent parameter given a leaf's depth.
float ResolveParam(const DepthDependentParam& param, int32 depth);

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_PARAMS_H_
