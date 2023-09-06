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
#include "tensorflow/core/tfrt/kernels/stream_ops.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/kernels/stream_ops_util.h"
#include "tensorflow/core/tfrt/runtime/stream.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace tfrt_stub {

namespace {
constexpr absl::string_view kBatchedStepIdName =
    "__streamed_result_batched_step_id";

bool AreNamesUnique(absl::Span<const std::string> names) {
  absl::flat_hash_set<absl::string_view> unique_names(names.begin(),
                                                      names.end());
  return unique_names.size() == names.size();
}

}  // namespace

PwStreamResultsOp::PwStreamResultsOp(tensorflow::OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("names", &names_));
  OP_REQUIRES(
      ctx, AreNamesUnique(names_),
      absl::InvalidArgumentError("Duplicate tensor names are not allowed"));

  // If this op is inside `tf.BatchFunction`, the `tf-pw-stream-result-batching`
  // pass inserts a special input indicating per-example step ids at the end.
  // This special input name must match that from the pass implementation.
  for (int i = 0; i < names_.size(); ++i) {
    if (names_[i] == kBatchedStepIdName) {
      OP_REQUIRES(ctx, i == names_.size() - 1,
                  absl::InternalError(
                      "step id must be the last input of `PwStreamResults`"));
      break;
    }
  }

  // The following are private properties that don't exist in SavedModel but
  // are populated dynamically by `ScopedStreamCallback::RewriteModule`.
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr("_controller_address", &controller_address_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("_model_name", &model_name_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("_callback_id", &callback_id_.id));

  auto interface = tensorflow::tfrt_stub::GetGlobalStreamInterfaceFactory()
                       .CreateWorkerStreamInterface()(controller_address_);
  OP_REQUIRES_OK(ctx, interface.status());
  stream_ = *std::move(interface);
}

void PwStreamResultsOp::Compute(tensorflow::OpKernelContext* ctx) {
  OP_REQUIRES(
      ctx, ctx->num_inputs() == names_.size(),
      absl::InvalidArgumentError(absl::StrCat(
          "The number of inputs (", ctx->num_inputs(),
          ") must match the number of tensor names (", names_.size(), ")")));

  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("PwStreamResults",
                                        {{"callback_id", callback_id_.id}});
  });

  absl::Cleanup latency_timer([this, start_time = absl::Now()]() {
    absl::Duration latency = absl::Now() - start_time;
    stream_->RecordSendLatency(model_name_, latency);
  });

  tensorflow::Tensor step_ids;
  bool has_step_id_inputs = false;
  if (names_.back() == kBatchedStepIdName) {
    // Use the explicitly provided step ids.
    step_ids = ctx->input(names_.size() - 1);
    has_step_id_inputs = true;
  } else {
    step_ids = tensorflow::Tensor(static_cast<int64_t>(ctx->step_id()));
  }

  std::vector<tensorflow::Tensor> batched_tensors;
  batched_tensors.resize(names_.size() - (has_step_id_inputs ? 1 : 0));
  for (int i = 0; i < batched_tensors.size(); ++i) {
    batched_tensors[i] = ctx->input(i);
  }

  auto responses = UnbatchStreamResults(step_ids, batched_tensors);
  OP_REQUIRES_OK(ctx, responses.status());

  OP_REQUIRES_OK(ctx, stream_->InvokeStreamCallback(callback_id_, names_,
                                                    responses.value()));
}

REGISTER_KERNEL_BUILDER(Name("PwStreamResults").Device(tensorflow::DEVICE_CPU),
                        PwStreamResultsOp);

}  // namespace tfrt_stub
}  // namespace tensorflow
