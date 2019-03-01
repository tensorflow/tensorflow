/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_NODE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_NODE_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

class ExecuteNode : public EagerNode {
 public:
  ExecuteNode(uint64 id, EagerContext* ctx, Device* op_device,
              const tensorflow::gtl::InlinedVector<TensorHandle*, 4>& inputs,
              KernelAndDevice* kernel, NodeExecStats* maybe_stats,
              StepStats* maybe_step_stats, GraphCollector* graph_collector,
              const DataTypeVector& output_dtypes,
              const tensorflow::gtl::InlinedVector<TensorHandle*, 2>& retvals)
      : EagerNode(id),
        ctx_(ctx),
        op_device_(op_device),
        inputs_(inputs),
        kernel_(kernel),
        maybe_stats_(maybe_stats),
        maybe_step_stats_(maybe_step_stats),
        graph_collector_(graph_collector),
        retvals_(retvals) {
    for (auto handle : inputs_) {
      handle->Ref();
    }
    for (auto handle : retvals_) {
      handle->Ref();
    }
  }

  ~ExecuteNode() override {
    for (auto handle : inputs_) {
      handle->Unref();
    }
    for (auto handle : retvals_) {
      handle->Unref();
    }
  }

  tensorflow::Status Run() override {
    const Status status = EagerKernelExecute(
        ctx_, op_device_, inputs_, kernel_, maybe_stats_.get(),
        maybe_step_stats_, graph_collector_, retvals_.begin(), retvals_.size());
    if (status.ok()) {
      return status;
    } else {
      return Status(status.code(),
                    strings::StrCat("Got error, \"", status.error_message(),
                                    "\" while executing kernel ",
                                    kernel_->kernel()->def().DebugString()));
    }
  }

 private:
  tensorflow::EagerContext* ctx_;
  tensorflow::Device* op_device_;
  tensorflow::gtl::InlinedVector<TensorHandle*, 4> inputs_;
  tensorflow::KernelAndDevice* kernel_;
  std::unique_ptr<NodeExecStats> maybe_stats_;
  StepStats* maybe_step_stats_;
  tensorflow::GraphCollector* graph_collector_;
  tensorflow::gtl::InlinedVector<TensorHandle*, 2> retvals_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_NODE_H_
