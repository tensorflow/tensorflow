/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

namespace {
using shape_inference::InferenceContext;

Status RemoteFusedGraphExecuteShapeFn(InferenceContext* c) {
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->UnknownShape());
  }
  return Status::OK();
}
}  // namespace

REGISTER_OP("RemoteFusedGraphExecute")
    .Input("inputs: Tinputs")
    .Output("outputs: Toutputs")
    .Attr("Tinputs: list(type) >= 0")
    .Attr("Toutputs: list(type) >= 0")
    .Attr("serialized_remote_fused_graph_execute_info: string")
    .SetShapeFn(RemoteFusedGraphExecuteShapeFn)
    .Doc(R"doc(
Execute a sub graph on a remote processor.

The graph specifications(such as graph itself, input tensors and output names)
are stored as a serialized protocol buffer of RemoteFusedGraphExecuteInfo
as serialized_remote_fused_graph_execute_info.
The specifications will be passed to a dedicated registered
remote fused graph executor.  The executor will send the graph specifications
to a remote processor and execute that graph.  The execution results
will be passed to consumer nodes as outputs of this node.

inputs: Arbitrary number of tensors with arbitrary data types
outputs: Arbitrary number of tensors with arbitrary data types
serialized_remote_fused_graph_execute_info: Serialized protocol buffer
of RemoteFusedGraphExecuteInfo which contains graph specifications.

)doc");

}  // namespace tensorflow
