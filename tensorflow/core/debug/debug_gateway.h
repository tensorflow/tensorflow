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

#ifndef TENSORFLOW_DEBUG_DEBUG_SESSION_H_
#define TENSORFLOW_DEBUG_DEBUG_SESSION_H_

#include <unordered_map>

#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/executor.h"

namespace tensorflow {

// Experimental. tfdb (TensorFlow Debugger): Gateway to intermediate node
// outputs during Session Run calls. Currently limited to DirectSession.
class DebugGateway {
 public:
  DebugGateway(DirectSession* session);
  virtual ~DebugGateway();

  // Callback for node completion. This callback is invoked only once for
  // a node regardless of whether it has one or more outputs. The value(s) of
  // the output tensor(s) are not necessarily available when this callback is
  // invoked. They may need to be asynchronously copied from device (e.g.,
  // GPU) to host, hence the need for the NodeValueCallback below.
  //
  // Args:
  //   node_name: Name of the node that has just completed execution
  //   any_output: Whether the node has any output(s)
  typedef std::function<void(const string& node_name, const bool any_output)>
      NodeCompletionCallback;
  void SetNodeCompletionCallback(NodeCompletionCallback callback);

  // Callback for node value. This is invoked when the value of a node's
  // output tensor is available on the host, possibly after copying from
  // a device (e.g., GPU).
  //
  // Args:
  //   node_name: Name of the node of which the output has become available
  //   output_slot: Output slot number of the output Tensor
  //   tensor_value: Reference to the tensor value
  //   is_ref: Whether the output of the reference type
  typedef std::function<void(const string& node_name, const int output_slot,
                             const Tensor& tensor_value, const bool is_ref)>
      NodeValueCallback;
  void SetNodeValueCallback(NodeValueCallback callback);

  // TODO(cais): Add whitelists for ops/tensors (e.g., {"A:0", "B:0"})
  // for node completion callback (whitelist_comp_) and node value callback
  // (whitelist_val_). If whitelist_comp_ is non-empty, the gateway will
  // invoke the NodeCompletionCallback only for the nodes specified in the
  // whitelist. And so forth for whitelist_val_.

 private:
  DirectSession* session_;
  // TODO(cais): DebugGateway currently supports only DirectSession. Add
  // support for GrpcSession.

  NodeCompletionCallback comp_cb_ = nullptr;
  NodeValueCallback val_cb_ = nullptr;

  typedef std::function<void(const Tensor* dst_tensor)> CopyDoneCallback;

  void CopyTensor(const string& node_name, const int output_slot,
                  const Tensor* src_tensor, OpKernelContext* ctx,
                  CopyDoneCallback copy_done_cb);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_DEBUG_DEBUG_SESSION_H_
