/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/kernels/ifrt_program_ops.h"

#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"

namespace tensorflow {
namespace tfrt_stub {

IfrtCallOp::IfrtCallOp(tensorflow::OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("program_id", &program_id_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr("variable_arg_indices", &variable_arg_indices_));
}

void IfrtCallOp::Compute(tensorflow::OpKernelContext* ctx) {
  absl::call_once(init_once_, [&]() {
    executable_ = tensorflow::ifrt_serving::ServingExecutableRegistry::Lookup(
        program_id_);
  });
  OP_REQUIRES(ctx, executable_ != nullptr,
              absl::NotFoundError(
                  absl::StrCat("Unknown program id '", program_id_, "'")));

  std::vector<Tensor> inputs;
  inputs.reserve(ctx->num_inputs());
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    inputs.push_back(ctx->input(i));
  }

  absl::StatusOr<std::vector<Tensor>> results =
      executable_->Execute(inputs, absl::MakeSpan(variable_arg_indices_));
  OP_REQUIRES(ctx, results.ok(), results.status());

  tensorflow::OpOutputList outputs(ctx, 0, results->size());
  for (int i = 0; i < results->size(); ++i) {
    outputs.set(i, (*results)[i]);
  }
}

REGISTER_KERNEL_BUILDER(Name("IfrtCall").Device(tensorflow::DEVICE_CPU),
                        IfrtCallOp);

}  // namespace tfrt_stub
}  // namespace tensorflow
