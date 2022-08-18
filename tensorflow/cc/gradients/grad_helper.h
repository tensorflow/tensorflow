/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CC_GRADIENTS_GRAD_HELPER_H_
#define TENSORFLOW_CC_GRADIENTS_GRAD_HELPER_H_

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"

namespace tensorflow {

// Helper function for reduction ops.
//
// input_shape: 1-D Tensor, the shape of the Tensor being reduced.
// axes: 1-D Tensor, the reduction axes.
//   Note that the reduction indices are in the range
//   -rank(input_shape), rank(input_shape)
// returns a 1-D Tensor, the output shape as if keep_dims were set to True.
Output ReducedShapeHelper(const Scope& scope, const Output& input_shape,
                          const Output& reduction_axes);

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_GRADIENTS_GRAD_HELPER_H_
