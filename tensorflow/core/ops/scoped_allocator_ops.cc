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
    .Attr("shape: shape")
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

'shapes' is a list of the shapes of the tensors that are to be allocated
by this ScopedAllocator.
'shape' is the shape of the output of this Op, i.e. the 1D backing tensor
from which the individual allocated tensors are aliased.
'sa_name' is the name assigned to the Node, for connectivity specification
and debugging.
'id' is a non-negative integer 'scope_id' handled by the ScopedAllocatorMgr.
'expected_call_count' is the number of individual tensors expected to
be allocated from the backing tensor.
)doc");

REGISTER_OP("_ScopedAllocatorConcat")
    .Output("output: T")
    .Input("backing: T")
    .Input("inputs: N * T")
    .Attr("shape: shape")
    .Attr("T: type")
    .Attr("reshape: bool = false")
    .Attr("sa_name: string")
    .Attr("id: int")
    .Attr("N: int >= 2")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape)
    .Doc(R"doc(
Acts like a Concat Op that merges multiple tensors into one, however it must
only be used in conjunction with a ScopedAllocator which is backing the memory
of all of its input tensors so that actually it just outputs a read-only
reference to that ScopedAllocator's backing tensor.

This is an experimental op for internal use only.  It is possible to use this
op in unsafe ways.

'backing' is the backing tensor, i.e. the output of an upstream ScopedAllocator.
'inputs' is a list of nominal input tensors, all of which must be aliases
to regions of the backing tensor.  These will be outputs of upstream nodes
that allocate their outputs from the same ScopedAllocator.
'shape' is the shape of the output, which will usually be the same shape as
the input backing tensor.
'reshape' is true iff the output shape is to be different from that of
the input backing tensor.
'sa_name' is the Node name of the upstream ScopedAllocator.
'id' is the scope_id identifying the upstream ScopedAllocator.
'N' is the number of nominal inputs to be concatenated.
)doc");

REGISTER_OP("_ScopedAllocatorSplit")
    .Output("output: N * T")
    .Input("concat: T")
    .Input("split: N * T")
    .Attr("T: type")
    .Attr("sa_name: string")
    .Attr("id: int")
    .Attr("N: int >= 2")
    .Attr("shapes: list(shape)")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShapes)
    .Doc(R"doc(
Acts roughly like a SplitV Op that splits one tensor into multiple tensors
but must only be used in conjunction with corresponding ScopedAllocator
and ScopedAllocatorConcat instances.  In practice it is provided as inputs
the backing tensor as first input, which contains the concatenated values,
and a list of alias tensors as its other input and it simply outputs that
second list.

This is an experimental op for internal use only.  It is possible to use this
op in unsafe ways.

'concat' is the single output produced by an upstream ScopedAllocatorConcat
node.  This is actually the backing tensor from a ScopedAllocator node
upstream of the ScopedAllocatorConcat.
'split' is a list of tensors aliased from the backing tensor.  It will
become the output of this ScopedAllocatorSplit node.
'type' is the common DataType of all of the input and output tensors.
'sa_name' is the Node name of the upstream ScopedAllocator.
'id' is the scope_id identifying the upstream ScopedAllocator.
'N' is the number of split tensors.
'shapes' is a list of the split tensor shapes.
)doc");

}  // end namespace tensorflow
