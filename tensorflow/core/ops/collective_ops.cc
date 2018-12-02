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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("CollectiveReduce")
    .Input("input: T")
    .Output("data: T")
    .Attr("T: {float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("merge_op: {'Min', 'Max', 'Mul', 'Add'}")
    .Attr("final_op: {'Id', 'Div'}")
    .Attr("subdiv_offsets: list(int)")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("CollectiveBcastSend")
    .Input("input: T")
    .Output("data: T")
    .Attr("T: {float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("CollectiveBcastRecv")
    .Output("data: T")
    .Attr("T: {float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

}  // namespace tensorflow
