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
// This file registers all TensorFlow Debugger (tfdbg) ops.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

// TensorFlow Debugger-inserted ops.
// These ops are used only internally by tfdbg. There is no API for users to
// direct create them. Users can create them indirectly by using
// RunOptions.debug_options during Session::Run() call. See tfdbg documentation
// for more details.
REGISTER_OP("Copy")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Attr("debug_ops_spec: list(string) = []")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Copy Op.

Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
device on which the tensor is allocated.
N.B.: If the all downstream attached debug ops are disabled given the current
gRPC gating status, the output will simply forward the input tensor without
deep-copying. See the documentation of Debug* ops for more details.

Unlike the CopyHost Op, this op does not have HostMemory constraint on its
input or output.

input: Input tensor.
output: Output tensor, deep-copied from input.
tensor_name: The name of the input tensor.
debug_ops_spec: A list of debug op spec (op, url, gated_grpc) for attached debug
  ops. Each element of the list has the format
  <debug_op>;<grpc_url>;<gated_grpc>, wherein gated_grpc is boolean represented
  as 0/1. E.g., "DebugIdentity;grpc://foo:3333;1",
  "DebugIdentity;file:///tmp/tfdbg_1;0".
)doc");

REGISTER_OP("CopyHost")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Attr("debug_ops_spec: list(string) = []")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Copy Host Op.

Performs CPU-to-CPU deep-copying of tensor.
N.B.: If the all downstream attached debug ops are disabled given the current
gRPC gating status, the output will simply forward the input tensor without
deep-copying. See the documentation of Debug* ops for more details.

Unlike the Copy Op, this op has HostMemory constraint on its input or output.

input: Input tensor.
output: Output tensor, deep-copied from input.
tensor_name: The name of the input tensor.
debug_ops_spec: A list of debug op spec (op, url, gated_grpc) for attached debug
  ops. Each element of the list has the format
  <debug_op>;<grpc_url>;<gated_grpc>, wherein gated_grpc is boolean represented
  as 0/1. E.g., "DebugIdentity;grpc://foo:3333;1",
  "DebugIdentity;file:///tmp/tfdbg_1;0".
)doc");

REGISTER_OP("DebugIdentity")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("device_name: string = ''")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .Attr("gated_grpc: bool = false")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Debug Identity Op.

Provides an identity mapping of the non-Ref type input tensor for debugging.

input: Input tensor, non-Reference type.
output: Output tensor that equals the input tensor.
tensor_name: Name of the input tensor.
debug_urls: List of URLs to debug targets, e.g.,
  file:///foo/tfdbg_dump, grpc:://localhost:11011
gated_grpc: Whether this op will be gated. If any of the debug_urls of this
  debug node is of the grpc:// scheme, when the value of this attribute is set
  to True, the data will not actually be sent via the grpc stream unless this
  debug op has been enabled at the debug_url. If all of the debug_urls of this
  debug node are of the grpc:// scheme and the debug op is enabled at none of
  them, the output will be an empty Tensor.
)doc");

REGISTER_OP("DebugNanCount")
    .Input("input: T")
    .Output("output: int64")  // The debug signal (nan count) is int64
    .Attr("T: type")
    .Attr("device_name: string = ''")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .Attr("gated_grpc: bool = false")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Debug NaN Value Counter Op

Counts number of NaNs in the input tensor, for debugging.

input: Input tensor, non-Reference type.
output: An integer output tensor that is the number of NaNs in the input.
tensor_name: Name of the input tensor.
debug_urls: List of URLs to debug targets, e.g.,
  file:///foo/tfdbg_dump, grpc:://localhost:11011.
gated_grpc: Whether this op will be gated. If any of the debug_urls of this
  debug node is of the grpc:// scheme, when the value of this attribute is set
  to True, the data will not actually be sent via the grpc stream unless this
  debug op has been enabled at the debug_url. If all of the debug_urls of this
  debug node are of the grpc:// scheme and the debug op is enabled at none of
  them, the output will be an empty Tensor.
)doc");

REGISTER_OP("DebugNumericSummary")
    .Input("input: T")
    .Output("output: double")
    .Attr("T: type")
    .Attr("device_name: string = ''")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .Attr("lower_bound: float = -inf")
    .Attr("upper_bound: float = inf")
    .Attr("mute_if_healthy: bool = false")
    .Attr("gated_grpc: bool = false")
    .SetAllowsUninitializedInput()
    // Note: this could return a more specific shape if needed in future.
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Debug Numeric Summary Op.

Provide a basic summary of numeric value types, range and distribution.

input: Input tensor, non-Reference type, float or double.
output: A double tensor of shape [14 + nDimensions], where nDimensions is the
  the number of dimensions of the tensor's shape. The elements of output are:
  [0]: is initialized (1.0) or not (0.0).
  [1]: total number of elements
  [2]: NaN element count
  [3]: generalized -inf count: elements <= lower_bound. lower_bound is -inf by
    default.
  [4]: negative element count (excluding -inf), if lower_bound is the default
    -inf. Otherwise, this is the count of elements > lower_bound and < 0.
  [5]: zero element count
  [6]: positive element count (excluding +inf), if upper_bound is the default
    -inf. Otherwise, this is the count of elements < upper_bound and > 0.
  [7]: generalized +inf count, elements >= upper_bound. upper_bound is +inf by
    default.
Output elements [1:8] are all zero, if the tensor is uninitialized.
  [8]: minimum of all non-inf and non-NaN elements.
       If uninitialized or no such element exists: +inf.
  [9]: maximum of all non-inf and non-NaN elements.
       If uninitialized or no such element exists: -inf.
  [10]: mean of all non-inf and non-NaN elements.
        If uninitialized or no such element exists: NaN.
  [11]: variance of all non-inf and non-NaN elements.
        If uninitialized or no such element exists: NaN.
  [12]: Data type of the tensor encoded as an enum integer. See the DataType
        proto for more details.
  [13]: Number of dimensions of the tensor (ndims).
  [14+]: Sizes of the dimensions.

tensor_name: Name of the input tensor.
debug_urls: List of URLs to debug targets, e.g.,
  file:///foo/tfdbg_dump, grpc:://localhost:11011
lower_bound: (float) The lower bound <= which values will be included in the
  generalized -inf count. Default: -inf.
upper_bound: (float) The upper bound >= which values will be included in the
  generalized +inf count. Default: +inf.
mute_if_healthy: (bool) Do not send data to the debug URLs unless at least one
  of elements [2], [3] and [7] (i.e., the nan count and the generalized -inf and
  inf counts) is non-zero.
gated_grpc: Whether this op will be gated. If any of the debug_urls of this
  debug node is of the grpc:// scheme, when the value of this attribute is set
  to True, the data will not actually be sent via the grpc stream unless this
  debug op has been enabled at the debug_url. If all of the debug_urls of this
  debug node are of the grpc:// scheme and the debug op is enabled at none of
  them, the output will be an empty Tensor.

)doc");

}  // namespace tensorflow
