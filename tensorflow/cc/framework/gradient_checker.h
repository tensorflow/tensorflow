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

#ifndef TENSORFLOW_CC_FRAMEWORK_GRADIENT_CHECKER_H_
#define TENSORFLOW_CC_FRAMEWORK_GRADIENT_CHECKER_H_

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

/// Returns in 'max_error' the maximum element-wise error for dy/dx between the
/// computed and numeric Jacobian matrices where 'xs' and 'ys' are tensors.
/// X_T and Y_T are the c++ types for the x and y tensors, and JAC_T is a
/// real-valued type to store the Jacobian derivatives dy/dx.
/// This function adds operations to the graph associated with 'scope'.
///
/// Examples:
/// if y = Square(x), where x (and so y) are DT_FLOAT,
/// <X_T, Y_T, JAC_T> should be <float, float, float>
///
/// if y = Square(x), where x (and so y) are DT_DOUBLE,
/// <X_T, Y_T, JAC_T> should be <double, double, double>
///
/// if y = Square(x), where x (and so y) are DT_COMPLEX64,
/// <X_T, Y_T, JAC_T> should be <complex64, complex64, float>
/// Note that JAC_T is always real-valued, and should be an appropriate
/// precision to host the partial derivatives for dy/dx
///
/// if y = ComplexAbs(x) where x is DT_COMPLEX64 (so y is DT_FLOAT)
/// <X_T, Y_T, JAC_T> should be <complex64, float, float>
///
/// if y = Complex(x, x) where x is DT_FLOAT (so y is DT_COMPLEX64)
/// <X_T, Y_T, JAC_T> should be <float, complex64, float>
template <typename X_T, typename Y_T, typename JAC_T>
Status ComputeGradientError(const Scope& scope, const OutputList& xs,
                            const std::vector<TensorShape>& x_shapes,
                            const OutputList& ys,
                            const std::vector<TensorShape>& y_shapes,
                            JAC_T* max_error);

/// Overload of ComputeGradientError which takes an initial value for 'x'.
template <typename X_T, typename Y_T, typename JAC_T>
Status ComputeGradientError(const Scope& scope, const Output& x,
                            const Tensor& x_init_value, const Output& y,
                            const TensorShape& y_shape, JAC_T* max_error);

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_GRADIENT_CHECKER_H_
