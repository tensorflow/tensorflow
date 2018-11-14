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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

REGISTER_OP("XRTAllocate")
    .Input("allocation: string")
    .Output("handle: int64")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Reads a literal proto and transfers it to TPU device memory.

'allocation' is a serialized xrt::TPUAllocation proto.
'handle' is an id that can be used in other ops to refer to the allocation.
)");

REGISTER_OP("XRTSubTuple")
    .Input("base_handle: int64")
    .Input("shape_index: int32")
    .Output("output_handle: int64")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Returns a handle to a sub-tuple of an allocated tuple.

'base_handle' is the id of the on-device allocation.
'shape_index' is a vector of integers describing an XLA ShapeIndex.
'output_handle' is an id that can be used in other ops to refer to the
sub-tuple.
)");

REGISTER_OP("XRTSubTupleAndRelease")
    .Input("base_handle: int64")
    .Input("shape_index: int32")
    .Output("output_handle: int64")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Returns a handle to a sub-tuple of an allocated tuple, and releases the handle
of the input tuple.

'base_handle' is the id of the on-device allocation.
'shape_index' is a vector of integers describing an XLA ShapeIndex.
'output_handle' is an id that can be used by other ops to refer to the
sub-tuple.
)");

REGISTER_OP("XRTMakeTuple")
    .Attr("Ninputs: int")
    .Input("tuple_description: string")
    .Input("input_handles: Ninputs * int64")
    .Output("output_handle: int64")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Returns a handle to a new allocation constructed by assembling existing
allocations in a tuple.

'tuple_description' is a serialized xrt::XLATupleNode proto describing the
shape of the output tuple, and whether each input handle should be aliased or
released.
'input_handles' is a list of input handles to assemble into the output tuple.
'output_handle' is an id that can be used by other ops to refer to the new
tuple.
'Ninputs' is the number of input handles.
)");

REGISTER_OP("XRTReadLiteral")
    .Input("handle: int64")
    .Output("literal: string")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Copies an allocated tuple from device memory and returns it as a literal.

'handle' is the id returned from the Op that produced the on-device allocation.
'literal' is a serialized xla::LiteralProto proto.
)");

REGISTER_OP("XRTReadLiteralAndRelease")
    .Input("handle: int64")
    .Output("literal: string")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Copies an allocated tuple from device memory, and returns it as a literal, and
releases the handle.

'handle' is the id returned from the Op that produced the on-device allocation.
'literal' is a serialized xla::LiteralProto proto.
)");

REGISTER_OP("XRTReleaseAllocationHandle")
    .Input("handle: int64")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(
        R"(
Discards an allocation from device memory. The handle cannot be subsequently
used.

'handle' is the id returned from the Op that produced the on-device allocation.
)");

}  // namespace tensorflow
