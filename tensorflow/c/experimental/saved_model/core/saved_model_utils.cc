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

#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"

#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/tf_tensor_internal.h"

namespace tensorflow {
namespace internal {

Status TensorProtoToConstant(ImmediateExecutionContext* ctx,
                             const TensorProto& proto,
                             std::unique_ptr<Constant>* output) {
  tensorflow::Tensor tensor;
  bool parse_result = tensor.FromProto(proto);
  if (!parse_result) {
    return errors::Internal("Failed to parse tensor from tensorproto");
  }

  TensorInterface tensor_interface(std::move(tensor));
  return Constant::Create(ctx, &tensor_interface, output);
}

}  // namespace internal
}  // namespace tensorflow
