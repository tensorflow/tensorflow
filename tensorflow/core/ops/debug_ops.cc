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
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("CopyHost")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Attr("debug_ops_spec: list(string) = []")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("DebugIdentity")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("device_name: string = ''")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .Attr("gated_grpc: bool = false")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("DebugNanCount")
    .Input("input: T")
    .Output("output: int64")  // The debug signal (nan count) is int64
    .Attr("T: type")
    .Attr("device_name: string = ''")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .Attr("gated_grpc: bool = false")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::ScalarShape);

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
    .SetShapeFn(shape_inference::UnknownShape);

// tfdbg v2.
REGISTER_OP("DebugIdentityV2")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tfdbg_context_id: string = ''")
    .Attr("op_name: string = ''")
    .Attr("output_slot: int = -1")
    .Attr("tensor_debug_mode: int = -1")
    .Attr("debug_urls: list(string) = []")
    .Attr("circular_buffer_size: int = 1000")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("DebugNumericSummaryV2")
    .Input("input: T")
    .Output("output: output_dtype")
    .Attr("output_dtype: {float32, float64} = DT_FLOAT")
    .Attr("T: type")
    .Attr("tensor_debug_mode: int = -1")
    .Attr("tensor_id: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);
}  // namespace tensorflow
