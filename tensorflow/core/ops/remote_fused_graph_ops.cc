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

// TODO(satok): Implement shape_inference
REGISTER_OP("RemoteFusedGraphExecute")
    .Input("values: M * T")
    .Output("output: N * U")
    .Attr("M: int >= 0")
    .Attr("N: int >= 0")
    .Attr("T: type")
    .Attr("U: type")
    .Attr("serialized_graph_transfer_info: string")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Execute a sub graph on a remote processor transferred by GraphTransferer.
The graph specifications are serialized by protobuf as graph_transfer_info.
The implementation / limitations may differ for each platform
and each available peripheral.
)doc");

}  // namespace tensorflow
