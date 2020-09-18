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

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gradients {

// Get a scalar TensorHandle with given value
Status ScalarTensorHandle(AbstractContext* ctx, float value,
                          AbstractTensorHandle** tensor);

// Get a TensorHandle with given float values and dimensions
Status TensorHandleWithDimsFloat(AbstractContext* ctx, float data[],
                                 int64_t dims[], int num_dims,
                                 AbstractTensorHandle** tensor);

// Get a TensorHandle with given int values and dimensions
Status TensorHandleWithDimsInt(AbstractContext* ctx, int data[], int64_t dims[],
                               int num_dims, AbstractTensorHandle** tensor);

// Places data from `t` into *result_tensor.
Status GetValue(AbstractTensorHandle* t, TF_Tensor** result_tensor);

// Util function that wraps an AbstractTensorHandle* with given data and dims.
AbstractTensorHandlePtr GetTensorHandleUtilFloat(AbstractContext* ctx,
                                                 float vals[], int64_t dims[],
                                                 int num_dims);

// Util function that wraps an AbstractTensorHandle* with given data and dims.
AbstractTensorHandlePtr GetTensorHandleUtilInt(AbstractContext* ctx, int vals[],
                                               int64_t dims[], int num_dims);

// Util function that wraps an AbstractTensorHandle* with given data.
AbstractTensorHandlePtr GetScalarTensorHandleUtil(AbstractContext* ctx,
                                                  float val);

// Performs gradient update for each weight using given learning rate.
Status UpdateWeights(AbstractContext* ctx,
                     std::vector<AbstractTensorHandle*>& grads,
                     std::vector<AbstractTensorHandle*>& weights,
                     AbstractTensorHandle* learning_rate);

using Model = std::function<Status(
    AbstractContext*, absl::Span<AbstractTensorHandle* const>,
    absl::Span<AbstractTensorHandle*>, const GradientRegistry&)>;

// Runs given model in either graph or eager mode depending on value of
// use_function.
Status RunModel(Model model, AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, bool use_function,
                const GradientRegistry& registry);

// Builds context and returns inside *ctx.
Status BuildImmediateExecutionContext(bool use_tfrt, AbstractContext** ctx);

}  // namespace gradients
}  // namespace tensorflow
