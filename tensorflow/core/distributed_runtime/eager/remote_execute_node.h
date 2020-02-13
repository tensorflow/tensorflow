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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_EXECUTE_NODE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_EXECUTE_NODE_H_

#include <cstddef>

#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/shape_inference.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {
namespace eager {

// RemoteExecuteNode is an implementation of EagerNode which enqueues
// an operation via RPC in a remote EagerService.
class RemoteExecuteNode : public AsyncEagerNode {
 public:
  RemoteExecuteNode(std::unique_ptr<EnqueueRequest> request, Device* device,
                    EagerClient* eager_client, const NodeDef& ndef,
                    FunctionLibraryDefinition* lib_def,
                    const gtl::InlinedVector<TensorHandle*, 4>& inputs,
                    absl::Span<TensorHandle*> retvals)
      : AsyncEagerNode(),
        request_(std::move(request)),
        device_(device),
        eager_client_(eager_client),
        ndef_(ndef),
        lib_def_(lib_def),
        inputs_(inputs) {
    // Copy the output handles, since the container for them might get
    // destroyed.
    for (auto handle : retvals) {
      handle->Ref();
      retvals_.push_back(handle);
    }

    // This is required to ensure that the tensor handles stay alive across the
    // execution.
    for (auto handle : inputs_) {
      handle->Ref();
    }
    eager_client_->Ref();
  }

  ~RemoteExecuteNode() override {
    for (auto handle : retvals_) {
      handle->Unref();
    }

    for (auto handle : inputs_) {
      handle->Unref();
    }
    eager_client_->Unref();
  }

  Status Prepare() override {
    return RunShapeInference(ndef_, *lib_def_, inputs_, retvals_);
  }

  void RunAsync(StatusCallback done) override;

  void Abort(Status status) override {
    for (auto handle : retvals_) {
      handle->Poison(status);
    }
  }

  string DebugString() const override {
    string out = "[RemoteExecuteNode]";
    strings::StrAppend(&out, " request: ", request_->DebugString());
    strings::StrAppend(&out, ", target_device: ", device_->name());
    return out;
  }

 private:
  std::unique_ptr<EnqueueRequest> request_;
  Device* device_;             // Not owned
  EagerClient* eager_client_;  // Not owned, and must outlive this node.
  const NodeDef ndef_;
  const FunctionLibraryDefinition* lib_def_;
  gtl::InlinedVector<TensorHandle*, 4> inputs_;
  gtl::InlinedVector<TensorHandle*, 2> retvals_;
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_EXECUTE_NODE_H_
