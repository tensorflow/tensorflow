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

namespace tensorflow {

REGISTER_OP("_ScopedAllocator")
    .Output("output: T")
    .Attr("shapes: list(shape)")
    .Attr("T: type")
    .Attr("sa_name: string")
    .Attr("id: int")
    .Attr("expected_call_count: int")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape)
    .Doc(R"doc(
Allocates a mutable tensor that becomes available to appropriately annotated
downstream Ops as backing store for their output tensor allocations via the
ScopedAllocatorMgr.
Returns a reference to this value.

This is an experimental op for internal use only.  It is possible to use this
op in unsafe ways.
)doc");

REGISTER_OP("_ScopedAllocatorConcat")
    .Output("output: T")
    .Input("backing: T")
    .Input("inputs: N * T")
    .Attr("shape: shape")
    .Attr("T: type")
    .Attr("sa_name: string")
    .Attr("id: int")
    .Attr("N: int >= 2")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape)
    .Doc(R"doc(
Acts like a Concat Op that merges multple tensors into one, however it must
only be used in conjunction with a ScopedAllocator which is backing the memory
of all of its input tensors so that actually it just outputs a read-only
reference to that ScopedAllocator's backing tensor.

This is an experimental op for internal use only.  It is possible to use this
op in unsafe ways.
)doc");

REGISTER_OP("_ScopedAllocatorSplit")
    .Output("output: N * T")
    .Input("concat: T")
    .Input("split: N * T")
    .Attr("T: type")
    .Attr("sa_name: string")
    .Attr("id: int")
    .Attr("N: int >= 2")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape)
    .Doc(R"doc(
Acts like a Concat Op that merges multple tensors into one, however it must
only be used in conjunction with a ScopedAllocator which is backing the memory
of all of its input tensors so that actually it just outputs a read-only
reference to that ScopedAllocator's backing tensor.

This is an experimental op for internal use only.  It is possible to use this
op in unsafe ways.
)doc");

}  // end namespace tensorflow
