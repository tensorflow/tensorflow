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

#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "tensorflow/core/tfrt/kernels/future_tensor_variant.h"

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

  absl::StatusOr<tsl::Future<std::vector<Tensor>>> results_future =
      executable_->Execute(inputs, absl::MakeSpan(variable_arg_indices_));
  OP_REQUIRES(ctx, results_future.ok(), results_future.status());

  // Determine the number of outputs from metadata if possible, or we might need
  // to wait if we don't know. But IfrtCallOp knows its number of outputs from
  // the NodeDef.
  int num_outputs = ctx->num_outputs();
  std::vector<tsl::Promise<Tensor>> promises;
  promises.reserve(num_outputs);

  for (int i = 0; i < num_outputs; ++i) {
    auto [promise, future] = tsl::MakePromise<Tensor>();
    promises.push_back(std::move(promise));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(i, TensorShape({}), &output_tensor));
    output_tensor->scalar<Variant>()() = FutureTensorVariant{std::move(future)};
  }

  results_future->OnReady(
      [promises = std::move(promises)](
          absl::StatusOr<std::vector<Tensor>> results_or) mutable {
        if (!results_or.ok()) {
          for (auto& promise : promises) {
            promise.Set(results_or.status());
          }

          return;
        }
        auto& results = *results_or;
        if (results.size() != promises.size()) {
          auto status = absl::InternalError(
              absl::StrCat("Expected ", promises.size(), " results, but got ",
                           results.size()));
          for (auto& promise : promises) {
            promise.Set(status);
          }
          return;
        }
        for (int i = 0; i < results.size(); ++i) {
          promises[i].Set(std::move(results[i]));
        }
      });
}

IfrtAwaitOp::IfrtAwaitOp(tensorflow::OpKernelConstruction* ctx)
    : AsyncOpKernel(ctx) {}

void IfrtAwaitOp::ComputeAsync(tensorflow::OpKernelContext* ctx,
                               DoneCallback done) {
  const Tensor& input_future_tensor = ctx->input(0);
  OP_REQUIRES_ASYNC(
      ctx, input_future_tensor.dtype() == DT_VARIANT,
      absl::InvalidArgumentError("Input must be a variant tensor"), done);
  const Variant& variant = input_future_tensor.scalar<Variant>()();
  const FutureTensorVariant* wrapper = variant.get<FutureTensorVariant>();
  OP_REQUIRES_ASYNC(ctx, wrapper != nullptr,
                    absl::InvalidArgumentError(
                        "Input variant does not contain a FutureTensorVariant"),
                    done);

  wrapper->future().OnReady([ctx, done](absl::StatusOr<Tensor> result) {
    if (!result.ok()) {
      ctx->SetStatus(result.status());
    } else {
      ctx->set_output(0, *result);
    }
    done();
  });
}

REGISTER_KERNEL_BUILDER(Name("IfrtCall").Device(tensorflow::DEVICE_CPU),
                        IfrtCallOp);
REGISTER_KERNEL_BUILDER(Name("IfrtCall").Device(tensorflow::DEVICE_TPU),
                        IfrtCallOp);
REGISTER_KERNEL_BUILDER(Name("IfrtAwait").Device(tensorflow::DEVICE_CPU),
                        IfrtAwaitOp);
REGISTER_KERNEL_BUILDER(Name("IfrtAwait").Device(tensorflow::DEVICE_TPU),
                        IfrtAwaitOp);

}  // namespace tfrt_stub
}  // namespace tensorflow
