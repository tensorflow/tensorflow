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
// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

vifrt.FuncV1 @good_custom_func(%arg0: !vifrt.control_v1) -> () {
  "vifrt.ReturnV1"() : () -> ()
} <{arg_attrs = [{}], res_attrs = [], sym_visibility = "vifrt.default"}> {vifrt.function}

// -----

// expected-error@+1 {{expected VIFRT or builtin types}}
vifrt.FuncV1 @custom_func_with_non_vifrt_type(%arg0: !ifrt.control) -> () {
  "vifrt.ReturnV1"() : () -> ()
} <{arg_attrs = [{}], res_attrs = [], sym_visibility = "vifrt.default"}> {vifrt.function}

// -----

module {
  // expected-error@+5 {{expected VIFRT or builtin types}}
  // expected-error@+4 {{failed to parse Vifrt_TypeAttrV1 parameter 'value' which is to be a `::mlir::Type`}}
  "vifrt.FuncV1"() ({
  ^bb0(%arg0: !ifrt.control):
    "vifrt.ReturnV1"() : () -> ()
  }) {arg_attrs = [{}], function_type = #vifrt.type_v1<!vifrt.func_v1<(!ifrt.control) -> ()>>, res_attrs = [], sym_name = "main", sym_visibility = "vifrt.default", vifrt.function} : () -> ()
}

// -----

// expected-error@+1 {{expected VIFRT or builtin type}}
module attributes {test.attr = #vifrt.type_v1<!ifrt.control>} {}

// -----

// expected-error@+1 {{Device list has negative logical id -1}}
module attributes {test.attr = #vifrt<devices_v1[-1, 1]>} {}

// -----

// expected-error@+1 {{Device list has duplicate logical id 0}}
module attributes {test.attr = #vifrt<devices_v1[0, 0]>} {}

// -----

// expected-error@+1 {{Out of range axis 2}}
module attributes {test.attr = #vifrt.sharding_param_v2<1x1 to [2] on 2>} {}

// -----

// expected-error@+1 {{`unreduced_axes` must contain non-negative values less than the number of mesh dimensions (=1). Saw: 2}}
module attributes {test.attr = #vifrt.sharding_param_v2<1x1 to [0] on 2 unreduced [2]>} {}

// -----

// expected-error@+1 {{start, end must be zero or positive}}
module attributes {test.attr = #vifrt<interval_v1[-1 : 1 : 1]>} {}

// -----

// expected-error@+1 {{step must be positive}}
module attributes {test.attr = #vifrt<interval_v1[0 : 1 : 0]>} {}

// -----

// expected-error@+1 {{interval is empty}}
module attributes {test.attr = #vifrt<interval_v1[2 : 1 : 1]>} {}

// -----

// expected-error@+1 {{from has 1 and to has 2, but they must have the same number of shards.}}
module attributes {test.attr = #vifrt.mapping_v1<[0 : 1 : 1] to [0 : 2 : 1]>} {}

// -----

// expected-error@+1 {{interval is empty}}
module attributes {test.attr = #vifrt<interval_v1[2 : 2 : 1]>} {}

// -----

// expected-error@+2 {{interval is empty}}
// expected-error@+1 {{failed to parse Vifrt_MappingAttrV1 parameter 'from_shards'}}
module attributes {test.attr = #vifrt.mapping_v1<[1 : 1 : 1] to [1 : 1 : 1]>} {}

// -----

// expected-error@+1 {{from has 2 and to has 4, but they must have the same number of shards.}}
module attributes {test.attr = #vifrt.mapping_v1<[0 : 3 : 2] to [0 : 4 : 1]>} {}

// -----

// expected-error@+1 {{in_array_index and out_array_index must be non-negative}}
module attributes {test.attr = #vifrt.array_mapping_v1<-1, 0, [#vifrt.mapping_v1<[0 : 1 : 1] to [0 : 1 : 1]>]>} {}

// -----

// expected-error@+1 {{in_array_index and out_array_index must be non-negative}}
module attributes {test.attr = #vifrt.array_mapping_v1<0, -1, [#vifrt.mapping_v1<[0 : 1 : 1] to [0 : 1 : 1]>]>} {}

// -----

// expected-error@+1 {{expected non-empty array of VifrtMappingV1Attr}}
module attributes {test.attr = #vifrt.array_mapping_v1<0, 0, []>} {}

// -----

func.func @array_v1_requires_same_dim_shards_rank() {
  // expected-error@+2 {{Requires dim shards to have the same rank as the array}}
  %0 = builtin.unrealized_conversion_cast to
      !vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v2<2 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  return
}

// -----

func.func @array_v1_invalid_layout_mode() {
  // expected-error@+2 {{Invalid layout mode}}
  %0 = builtin.unrealized_conversion_cast to
      !vifrt.array_v1<tensor<2x4xi32>, #vifrt.sharding_param_v2<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "invalid_layout">
  return
}

// -----

func.func @array_v1_non_scalar_token_shape() {
  // expected-error@+2 {{Token type cannot have non-empty shape}}
  %0 = builtin.unrealized_conversion_cast to
      !vifrt.array_v1<tensor<2x!vifrt.token_v1>, #vifrt.sharding_param_v2<1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  return
}

// -----

func.func @array_v1_non_vifrt_sharding_attr() {
  // expected-error@+2 {{expected VIFRT sharding attribute}}
  %0 = builtin.unrealized_conversion_cast to
      !vifrt.array_v1<tensor<2x4xi32>, 42, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  return
}

// -----

func.func @array_v1_unexpected_tensor_encoding() {
  // expected-error@+2 {{Unexpected tensor encoding on array shape}}
  %0 = builtin.unrealized_conversion_cast to
      !vifrt.array_v1<tensor<2x4xi32, "enc">, #vifrt.sharding_param_v2<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  return
}

// -----

func.func @array_v1_unsupported_element_type() {
  // expected-error@+2 {{Unsupported array element type}}
  %0 = builtin.unrealized_conversion_cast to
      !vifrt.array_v1<tensor<2x!vifrt.control_v1>, #vifrt.sharding_param_v2<1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  return
}

// -----

func.func @array_v1_shape_element_count_overflow() {
  // expected-error@+2 {{Array shape element count overflows 64-bit signed integer}}
  %0 = builtin.unrealized_conversion_cast to
      !vifrt.array_v1<tensor<4611686018427387904x4xi32>, #vifrt.sharding_param_v2<1x1 to [0] on 2>, [0, 1], memory_kind = "vifrt.default", layout = "vifrt.default">
  return
}

// -----

// expected-error@+1 {{unknown vifrt attribute: unknown_attr_v1}}
module attributes {test.attr = #vifrt.unknown_attr_v1} {}

// -----

func.func @unknown_vifrt_type() {
  // expected-error@+2 {{unknown vifrt type: unknown_type_v1}}
  %0 = builtin.unrealized_conversion_cast to
      !vifrt.unknown_type_v1
  return
}
