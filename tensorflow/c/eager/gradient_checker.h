/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/gradients_util.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace gradients {

using Model = std::function<Status(
    AbstractContext*, absl::Span<AbstractTensorHandle* const>,
    absl::Span<AbstractTensorHandle*>, const GradientRegistry&)>;

/** Returns numerical grad inside `dtheta_approx` given `forward` model and
 * parameter specified by `input_index`.
 *
 * I.e. if y = <output of the forward model> and w = inputs[input_index],
 * this will calculate dy/dw numerically.
 *
 * `use_function` indicates whether to use graph mode(true) or eager(false).
 *
 * `is_scalar_out` should be true when `forward` returns a scalar TensorHandle;
 *  else this function will reduce_sum the tensor to get a scalar to estimate
 *  the gradient with. Default is false.
 */
Status CalcNumericalGrad(AbstractContext* ctx, Model forward,
                         std::vector<AbstractTensorHandle*> inputs,
                         float* dtheta_approx, int input_index,
                         bool use_function, bool is_scalar_out = false);

}  // namespace gradients
}  // namespace tensorflow
