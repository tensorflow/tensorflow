/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
#define TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_

#include <functional>
#include <memory>

#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// Contains request related info.
struct RequestInfo {
  tfrt::RCReference<tfrt::RequestContext> tfrt_request_context;
  std::unique_ptr<WorkQueueInterface> request_queue;
  std::function<void(std::function<void()>)> runner;
};

// Creates a `RequestInfo` given relative data.
StatusOr<std::unique_ptr<RequestInfo>> SetUpRequestContext(
    const GraphExecutionRunOptions& run_options,
    const SessionMetadata& model_metadata, tfrt::HostContext* host,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    tfrt::ResourceContext* resource_context,
    const FallbackState& fallback_state);

// Runs on a function given input/output and other info.
tensorflow::Status GraphExecutionRunOnFunction(
    const GraphExecutionOptions& options,
    const GraphExecutionRunOptions& run_options,
    absl::string_view signature_name, const tfrt::Function& func,
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const tensorflow::Tensor> captures,
    std::vector<tensorflow::Tensor>* outputs,
    tfrt::ResourceContext* resource_context, const Runtime& runtime,
    const FallbackState& fallback_state,
    tfrt::RequestDeadlineTracker& req_deadline_tracker);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
