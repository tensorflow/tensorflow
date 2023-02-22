// RUN: xla-runtime-opt %s | FileCheck %s

// CHECK-LABEL: func @custom_arg(
// CHECK:  %[[ARG:.*]]: !testlib.custom_arg
func.func @custom_arg(%arg0: !testlib.custom_arg) {
  return
}

// CHECK-LABEL: func @enum(
// CHECK: enum = #testlib.enum_type<Baz>
func.func @enum() attributes { enum = #testlib.enum_type<Baz> } {
  return
}

// CHECK-LABEL: func @another_enum(
// CHECK: enum = #testlib.another_enum_type<Bar>
func.func @another_enum() attributes { enum = #testlib.another_enum_type<Bar> }
{
  return
}

// CHECK-LABEL: func @dims(
// CHECK: dims = #testlib.pair_of_dims<2, [1, 1], [2, 2]>
func.func @dims() attributes { dims = #testlib.pair_of_dims<2, [1, 1], [2, 2]> }
{
  return
}
