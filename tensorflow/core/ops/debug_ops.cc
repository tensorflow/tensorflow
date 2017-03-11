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

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

// EXPERIMENTAL: tfdbg debugger-inserted ops.
REGISTER_OP("Copy")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .SetAllowsUninitializedInput()
    .Doc(R"doc(
Copy Op.

Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
device on which the tensor is allocated.

Unlike the CopyHost Op, this op does not have HostMemory constraint on its
input or output.

input: Input tensor.
output: Output tensor, deep-copied from input.
tensor_name: The name of the input tensor.
)doc");

REGISTER_OP("CopyHost")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .SetAllowsUninitializedInput()
    .Doc(R"doc(
Copy Host Op.

Performs CPU-to-CPU deep-copying of tensor.

Unlike the Copy Op, this op has HostMemory constraint on its input or output.

input: Input tensor.
output: Output tensor, deep-copied from input.
tensor_name: The name of the input tensor.
)doc");

REGISTER_OP("DebugIdentity")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .SetAllowsUninitializedInput()
    .Doc(R"doc(
Debug Identity Op.

Provides an identity mapping of the non-Ref type input tensor for debugging.

input: Input tensor, non-Reference type.
output: Output tensor that equals the input tensor.
tensor_name: Name of the input tensor.
debug_urls: List of URLs to debug targets, e.g.,
            file:///foo/tfdbg_dump, grpc:://localhost:11011
)doc");

REGISTER_OP("DebugNanCount")
    .Input("input: T")
    .Output("output: int64")  // The debug signal (nan count) is int64
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .SetAllowsUninitializedInput()
    .Doc(R"doc(
Debug NaN Value Counter Op

Counts number of NaNs in the input tensor, for debugging.

input: Input tensor, non-Reference type.
output: An integer output tensor that is the number of NaNs in the input.
tensor_name: Name of the input tensor.
debug_urls: List of URLs to debug targets, e.g.,
            file:///foo/tfdbg_dump, grpc:://localhost:11011
)doc");

REGISTER_OP("DebugNumericSummary")
    .Input("input: T")
    .Output("output: double")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .SetAllowsUninitializedInput()
    .Doc(R"doc(
Debug Numeric Summary Op.

Provide a basic summary of numeric value types, range and distribution.

input: Input tensor, non-Reference type, float or double.
output: A double tensor of shape [12], the elements of which are:
  [0]: is initialized (1.0) or not (0.0).
  [1]: total number of elements
  [2]: -inf count
  [3]: negative element count (excluding -inf)
  [4]: zero element count
  [5]: positive element count (excluding +inf)
  [6]: +inf element count
  [7]: NaN element count
Output elements [1:8] are all zero, if the tensor is uninitialized.
  [8]: minimum of all non-inf and non-NaN elements.
       If uninitialized or no such element exists: +inf.
  [9]: maximum of all non-inf and non-NaN elements.
       If uninitialized or no such element exists: -inf.
  [10]: mean of all non-inf and non-NaN elements.
        If uninitialized or no such element exists: NaN.
  [11]: variance of all non-inf and non-NaN elements.
        If uninitialized or no such element exists: NaN.

tensor_name: Name of the input tensor.
debug_urls: List of URLs to debug targets, e.g.,
            file:///foo/tfdbg_dump, grpc:://localhost:11011
)doc");

}  // namespace tensorflow