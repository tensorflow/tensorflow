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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

Status RpcShapeOp(InferenceContext* c, bool try_rpc) {
  ShapeHandle address;
  ShapeHandle method;
  ShapeHandle request;
  ShapeHandle output;
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &address));
  if (c->Rank(address) == 1) {
    TF_RETURN_IF_ERROR(c->Merge(output, address, &output));
  }
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &method));
  if (c->Rank(method) == 1) {
    TF_RETURN_IF_ERROR(c->Merge(output, method, &output));
  }
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &request));
  if (c->Rank(request) == 1) {
    TF_RETURN_IF_ERROR(c->Merge(output, request, &output));
  }
  if (!c->RankKnown(output)) {
    output = request;
  }
  c->set_output(0, output);  // response
  if (try_rpc) {
    c->set_output(1, output);  // status_code
    c->set_output(2, output);  // status_message
  }
  return Status::OK();
}

REGISTER_OP("Rpc")
    .Input("address: string")
    .Input("method: string")
    .Input("request: string")
    .Attr("protocol: string = ''")
    .Attr("fail_fast: bool = true")
    .Attr("timeout_in_ms: int = 0")
    .Output("response: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      return RpcShapeOp(c, /*try_rpc=*/false);
    });

REGISTER_OP("TryRpc")
    .Input("address: string")
    .Input("method: string")
    .Input("request: string")
    .Attr("protocol: string = ''")
    .Attr("fail_fast: bool = true")
    .Attr("timeout_in_ms: int = 0")
    .Output("response: string")
    .Output("status_code: int32")
    .Output("status_message: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      return RpcShapeOp(c, /*try_rpc=*/true);
    });

}  // namespace tensorflow
