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
Reads a literal proto and transfers it to device memory.

'allocation' is a serialized xrt::XLAAllocation proto.
'handle' is an id that can be used in other ops to refer to the allocation.
)");

REGISTER_OP("XRTAllocateUninitialized")
    .Output("handle: int64")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Allocates a tensor to hold the specified shape in device memory.  The values
in the tensor are left uninitialized.

shape: The shapes which the tensor should have on device.

handle: An id that can be used in other ops to refer to the allocation.
)");

REGISTER_OP("XRTAllocateFromTensor")
    .Input("inputs: dtypes")
    .Output("handle: int64")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .Attr("layouts: list(int) = []")
    .Attr("make_tuple: bool = false")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Reads a list of tensors with optional layouts, and transfers it to device
memory.

inputs: The tensors holding the input data.
shapes: The shapes which the tensors should have on device. The i-th shape
corresponds to the i-th input. The shapes, together with the (optional)
layouts, helps creating the fully qualified shape of the data on the device.
The shapes can differ from the corresponding input one, as long as the total
number of elements matches. In other words, it is possible to feed an input
tensor with shape {8} and have a corresponding shape {2,2,2}.
layouts: A vector holding the requested layout in minor-to-major sequence.
If empty, the default layout will be used.
For a tuple, the layouts vector holds a linearized minor-to-major numbers
for all the tuple leaves, in the order they appear within the tuple.
The elements within the layouts sequence corresponding to a given tuple
subshape can be set to -1, to leave such subshape to the default shape.
handle: An id that can be used in other ops to refer to the allocation.
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

REGISTER_OP("XRTWriteLiteral")
    .Input("handle: int64")
    .Input("literal: string")
    .Output("output_handle: int64")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Copies the input literal into the device memory pointed to by handle.
Returns the handle itself.

'handle' is the id returned from the Op that produced the on-device allocation.
'literal' is a serialized xla::LiteralProto proto to be written to device memory.
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

REGISTER_OP("XRTReadToTensor")
    .Input("handles: int64")
    .Attr("release_handles: bool = False")
    .Attr("dtypes: list(type)")
    .Output("tensors: dtypes")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .Doc(
        R"(
Copies allocated values from device memory and returns them as zero or more
Tensors. If a handle refers to a non-tuple buffer, a single tensor is returned.
In general, the tensors returned for a handle correspond to an in-order traversal
of a the tuple-tree value referenced by the handle.

'handles' contains ids returned from Ops that produced on-device allocations.
At present, only a single (scalar) handle is supported.
'dtypes' are the expected types for each `Tensor` to be returned. If the
expected and actual tensor types do not match, an error is returned.
'release_handles': if True, `handles` are released.
'tensors' are the output Tensors.
)");

REGISTER_OP("XRTReleaseAllocationHandle")
    .Input("handle: int64")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(
        R"(
Discards one or more device memory handles. The handle(s) cannot be subsequently
used.

'handle' is the ID (or a vector of IDs) returned from the Op that produced the
on-device allocation.
)");

REGISTER_OP("XRTReleaseAllAllocations")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(
        R"(
Discards all the XRT allocations. All the client held handles will be invalid.
)");

REGISTER_OP("XRTCompactAllocations")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(
        R"(
Runs a device memory compaction cycle. This copies the device data behind the
currently alive allocation handles into host memory, releases the device memory
backing the handles, and re-allocate and send back the data to the device.
This operation helps with device memory fragmentation.
)");

}  // namespace tensorflow
