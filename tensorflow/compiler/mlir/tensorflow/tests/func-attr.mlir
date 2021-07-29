// RUN: tf-opt %s | tf-opt | FileCheck %s

// CHECK-LABEL: func @func_attr
// CHECK-SAME: tf._implements = #tf_type.func<@symbol_a, {attr0 = 1 : i32, attr1 = "random"}>
func @func_attr() attributes {tf._implements = #tf_type.func<@symbol_a, {attr0 = 1 : i32, attr1 = "random"}>} {
  return
}

// CHECK-LABEL: func @nested_func_attr
// CHECK-SAME: tf._implements = #tf_type.func<@symbol_a, {attr0 = 1 : i32, attr1 = "random", nested = #tf_type.func<@symbol_b, {attr2 = true, attr3 = 8.000000e+00 : f32}>}>
func @nested_func_attr() attributes {tf._implements = #tf_type.func<@symbol_a, {attr0 = 1 : i32, attr1 = "random", nested = #tf_type.func<@symbol_b, {attr2 = true, attr3 = 8.0 : f32}>}>} {
  return
}
