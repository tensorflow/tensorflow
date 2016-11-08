/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_GRADIENT_CHECKER_H_
#define THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_GRADIENT_CHECKER_H_

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

// Returns in 'max_error' the maximum element-wise error for dy/dx between the
// computed and numeric Jacobian matrices where 'xs' and 'ys' are tensors.
// This function adds operations to the graph associated with 'scope'.
template <typename T>
Status ComputeGradientError(const Scope& scope, const ops::OutputList& xs,
                            const std::vector<TensorShape>& x_shapes,
                            const ops::OutputList& ys,
                            const std::vector<TensorShape>& y_shapes,
                            T* max_error);

// Overload of ComputeGradientError which takes an initial value for 'x'.
template <typename T>
Status ComputeGradientError(const Scope& scope, const ops::Output& x,
                            const Tensor& x_init_value, const ops::Output& y,
                            const TensorShape& y_shape, T* max_error);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_GRADIENT_CHECKER_H_
