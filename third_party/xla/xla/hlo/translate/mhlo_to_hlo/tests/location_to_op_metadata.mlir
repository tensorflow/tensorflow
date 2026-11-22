// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(unknown)
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-NOT: metadata

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("AfterAll")
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="AfterAll"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("name@function")
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="name"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("file_name":2:8)
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={source_file="file_name" source_line=2 source_end_line=2 source_column=8 source_end_column=8}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("name(with)[]")
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="name(with)[]"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("name(anothername)"("file_name":2:8))
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="name(anothername)" stack_frame_id=1}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(fused["fused/location/file", "source.txt":42:5])
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="fused/location/file" source_file="source.txt" source_line=42 source_end_line=42 source_column=5 source_end_column=5}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(fused["name1", fused["nested_fusion":5:42, "name2"]])
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="name1;name2" source_file="nested_fusion" source_line=5 source_end_line=5 source_column=42 source_end_column=42}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(fused["multiple_sources", "source1":1:2, "source2":3:4])
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="multiple_sources" source_file="source2" source_line=3 source_end_line=3 source_column=4 source_end_column=4}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: tensor<4xf32> loc("x"), %arg1: tensor<4xf32> loc("y")) -> tensor<4xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK: parameter(0)
// CHECK-SAME: metadata={op_name="x"}
// CHECK: parameter(1)
// CHECK-SAME: metadata={op_name="y"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(#type_loc)
  func.return %0 : !mhlo.token
}
#name_loc = loc("aname")
#type_loc = loc("atype:"(#name_loc))

// CHECK: after-all
// CHECK-SAME: metadata={op_type="atype" op_name="aname"}

// -----

// CHECK-LABEL: %main
module @m {
  func.func public @main(%arg0: tensor<f64>) -> (tensor<f64>) {
    %0 = call @foo(%arg0) : (tensor<f64>) -> tensor<f64> loc("x")
    return %0 : tensor<f64>
  }
  func.func private @foo(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = mhlo.cosine %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
}

// CHECK: call({{.*}}), to_apply=%foo.{{[0-9.]+}}, metadata={op_name="x"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32> loc("my_add")
  func.return %0 : tensor<1x2xf32>
}

// CHECK: my_add{{.*}}, metadata={op_name="my_add"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: tensor<2xf32> loc("inputs.x"), %arg1: tensor<2xf32> loc("inputs.y")) -> tensor<2xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32> loc("my_add")
  func.return %0 : tensor<2xf32>
}

// CHECK: inputs_x{{.*}}, metadata={op_name="inputs.x"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: tensor<2xf32> loc("Arg_0.1"), %arg1: tensor<2xf32> loc("Arg_1.1")) -> tensor<2xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32> loc("my_add")
  func.return %0 : tensor<2xf32>
}

// CHECK: Arg_0{{.*}}, metadata={op_name="Arg_0.1"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) {mhlo.frontend_attributes = {xla_metadata_payload = "my_tokamax_payload:json_data", some_other_attr = "keep_me"}} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: add
// CHECK-SAME: metadata_payload="my_tokamax_payload:json_data"

// -----

// Two levels of MLIR inlining: the call sites' names compose outermost first,
// the op type comes from the inlined op, and all three frames are exported.

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(#inlined_loc)
  func.return %0 : !mhlo.token
}
#inner_frame = loc("inner_helper"("kernel.py":10:5))
#mid_frame = loc("mid_helper"("kernel.py":15:3))
#outer_frame = loc("outer_op"("kernel.py":20:1))
#callee_loc = loc("pallas_call:"("pallas_call"(#inner_frame)))
#mid_caller_loc = loc("call:"("first_call/jit(inner_helper)"(#mid_frame)))
#outer_caller_loc = loc("call:"("jit(outer_op)/jit(mid_helper)"(#outer_frame)))
#nested_caller_loc = loc(callsite(#mid_caller_loc at #outer_caller_loc))
#inlined_loc = loc(callsite(#callee_loc at #nested_caller_loc))

// CHECK: after-all
// CHECK-SAME: metadata={op_type="pallas_call" op_name="jit(outer_op)/jit(mid_helper)/first_call/jit(inner_helper)/pallas_call" stack_frame_id=3}

// -----

// When the call carried no name scope, its location is a bare traceback whose
// frames must not leak into the op name.

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(#jax_inlined_unscoped)
  func.return %0 : !mhlo.token
}
#unscoped_callee_frame = loc("inner_fn"("add.py":10:5))
#unscoped_caller_frame = loc("<module>"("add.py":20:3))
#unscoped_callee_op = loc("add:"("add"(#unscoped_callee_frame)))
#unscoped_caller_traceback = loc(callsite(#unscoped_callee_frame at #unscoped_caller_frame))
#jax_inlined_unscoped = loc(callsite(#unscoped_callee_op at #unscoped_caller_traceback))

// CHECK: after-all
// CHECK-SAME: metadata={op_type="add" op_name="add" stack_frame_id=3}

// -----

// MLIR uniques locations, so nested locations can share sub-locations. Each
// alias below uses the previous one four times: 41 locations that expand to
// 4^20 op names. The export must not visit all of them.

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token loc(#callsite20)) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(#callsite20)
  func.return %0 : !mhlo.token
}
#callsite0 = loc("x")
#callsite1 = loc(callsite(callsite(#callsite0 at #callsite0) at callsite(#callsite0 at #callsite0)))
#callsite2 = loc(callsite(callsite(#callsite1 at #callsite1) at callsite(#callsite1 at #callsite1)))
#callsite3 = loc(callsite(callsite(#callsite2 at #callsite2) at callsite(#callsite2 at #callsite2)))
#callsite4 = loc(callsite(callsite(#callsite3 at #callsite3) at callsite(#callsite3 at #callsite3)))
#callsite5 = loc(callsite(callsite(#callsite4 at #callsite4) at callsite(#callsite4 at #callsite4)))
#callsite6 = loc(callsite(callsite(#callsite5 at #callsite5) at callsite(#callsite5 at #callsite5)))
#callsite7 = loc(callsite(callsite(#callsite6 at #callsite6) at callsite(#callsite6 at #callsite6)))
#callsite8 = loc(callsite(callsite(#callsite7 at #callsite7) at callsite(#callsite7 at #callsite7)))
#callsite9 = loc(callsite(callsite(#callsite8 at #callsite8) at callsite(#callsite8 at #callsite8)))
#callsite10 = loc(callsite(callsite(#callsite9 at #callsite9) at callsite(#callsite9 at #callsite9)))
#callsite11 = loc(callsite(callsite(#callsite10 at #callsite10) at callsite(#callsite10 at #callsite10)))
#callsite12 = loc(callsite(callsite(#callsite11 at #callsite11) at callsite(#callsite11 at #callsite11)))
#callsite13 = loc(callsite(callsite(#callsite12 at #callsite12) at callsite(#callsite12 at #callsite12)))
#callsite14 = loc(callsite(callsite(#callsite13 at #callsite13) at callsite(#callsite13 at #callsite13)))
#callsite15 = loc(callsite(callsite(#callsite14 at #callsite14) at callsite(#callsite14 at #callsite14)))
#callsite16 = loc(callsite(callsite(#callsite15 at #callsite15) at callsite(#callsite15 at #callsite15)))
#callsite17 = loc(callsite(callsite(#callsite16 at #callsite16) at callsite(#callsite16 at #callsite16)))
#callsite18 = loc(callsite(callsite(#callsite17 at #callsite17) at callsite(#callsite17 at #callsite17)))
#callsite19 = loc(callsite(callsite(#callsite18 at #callsite18) at callsite(#callsite18 at #callsite18)))
#callsite20 = loc(callsite(callsite(#callsite19 at #callsite19) at callsite(#callsite19 at #callsite19)))

// CHECK: parameter(0)
// CHECK-SAME: metadata={op_name="x/{{[x/]*}}x"}
// CHECK: after-all
// CHECK-SAME: metadata={op_name="x/{{[x/]*}}x"}

// -----

// Same as above with FusedLoc: 101 locations that expand to 4^20 source
// locations, none of which holds an op name or a stack frame.

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(#fused20)
  func.return %0 : !mhlo.token
}
#fused0 = loc("source.txt":42:5)
#fused1 = loc(fused[fused<"a">[#fused0], fused<"b">[#fused0], fused<"c">[#fused0], fused<"d">[#fused0]])
#fused2 = loc(fused[fused<"a">[#fused1], fused<"b">[#fused1], fused<"c">[#fused1], fused<"d">[#fused1]])
#fused3 = loc(fused[fused<"a">[#fused2], fused<"b">[#fused2], fused<"c">[#fused2], fused<"d">[#fused2]])
#fused4 = loc(fused[fused<"a">[#fused3], fused<"b">[#fused3], fused<"c">[#fused3], fused<"d">[#fused3]])
#fused5 = loc(fused[fused<"a">[#fused4], fused<"b">[#fused4], fused<"c">[#fused4], fused<"d">[#fused4]])
#fused6 = loc(fused[fused<"a">[#fused5], fused<"b">[#fused5], fused<"c">[#fused5], fused<"d">[#fused5]])
#fused7 = loc(fused[fused<"a">[#fused6], fused<"b">[#fused6], fused<"c">[#fused6], fused<"d">[#fused6]])
#fused8 = loc(fused[fused<"a">[#fused7], fused<"b">[#fused7], fused<"c">[#fused7], fused<"d">[#fused7]])
#fused9 = loc(fused[fused<"a">[#fused8], fused<"b">[#fused8], fused<"c">[#fused8], fused<"d">[#fused8]])
#fused10 = loc(fused[fused<"a">[#fused9], fused<"b">[#fused9], fused<"c">[#fused9], fused<"d">[#fused9]])
#fused11 = loc(fused[fused<"a">[#fused10], fused<"b">[#fused10], fused<"c">[#fused10], fused<"d">[#fused10]])
#fused12 = loc(fused[fused<"a">[#fused11], fused<"b">[#fused11], fused<"c">[#fused11], fused<"d">[#fused11]])
#fused13 = loc(fused[fused<"a">[#fused12], fused<"b">[#fused12], fused<"c">[#fused12], fused<"d">[#fused12]])
#fused14 = loc(fused[fused<"a">[#fused13], fused<"b">[#fused13], fused<"c">[#fused13], fused<"d">[#fused13]])
#fused15 = loc(fused[fused<"a">[#fused14], fused<"b">[#fused14], fused<"c">[#fused14], fused<"d">[#fused14]])
#fused16 = loc(fused[fused<"a">[#fused15], fused<"b">[#fused15], fused<"c">[#fused15], fused<"d">[#fused15]])
#fused17 = loc(fused[fused<"a">[#fused16], fused<"b">[#fused16], fused<"c">[#fused16], fused<"d">[#fused16]])
#fused18 = loc(fused[fused<"a">[#fused17], fused<"b">[#fused17], fused<"c">[#fused17], fused<"d">[#fused17]])
#fused19 = loc(fused[fused<"a">[#fused18], fused<"b">[#fused18], fused<"c">[#fused18], fused<"d">[#fused18]])
#fused20 = loc(fused[fused<"a">[#fused19], fused<"b">[#fused19], fused<"c">[#fused19], fused<"d">[#fused19]])

// CHECK: after-all
// CHECK-SAME: metadata={source_file="source.txt" source_line=42 source_end_line=42 source_column=5 source_end_column=5}
