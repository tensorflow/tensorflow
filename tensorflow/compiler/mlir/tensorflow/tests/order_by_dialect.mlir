// RUN: tf-opt %s -allow-unregistered-dialect --tf-order-by-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: @interleave
func.func @interleave(%arg0: f32) -> (f32, f32, f32) attributes {ignore_side_effects_for_testing} {
  %0 = "x.a"(%arg0) : (f32) -> f32
  %1 = "y.a"(%arg0) : (f32) -> f32
  %2 = "z.a"(%arg0) : (f32) -> f32
  %3 = "x.b"(%0) : (f32) -> f32
  %4 = "y.b"(%1) : (f32) -> f32
  %5 = "z.b"(%2) : (f32) -> f32
  %6 = "x.c"(%3) : (f32) -> f32
  %7 = "y.c"(%4) : (f32) -> f32
  %8 = "z.c"(%5) : (f32) -> f32
  func.return %6, %7, %8 : f32, f32, f32
}
// CHECK: x.a
// CHECK-NEXT: x.b
// CHECK-NEXT: x.c
// CHECK-NEXT: y.a
// CHECK-NEXT: y.b
// CHECK-NEXT: y.c
// CHECK-NEXT: z.a
// CHECK-NEXT: z.b
// CHECK-NEXT: z.c

// -----

// CHECK-LABEL: @terminator
func.func @terminator(%arg0: f32) -> (f32) attributes {ignore_side_effects_for_testing} {
  func.call @terminator(%arg0) : (f32) -> (f32)
  "x.a"(%arg0) : (f32) -> ()
  "y.a"(%arg0) : (f32) -> ()
  "z.a"(%arg0) : (f32) -> ()
  func.return %arg0 : f32
}
// CHECK: x.a
// CHECK: y.a
// CHECK: z.a
// CHECK: return
// CHECK-NOT: call
// CHECK: }

// -----

// CHECK-LABEL: @fanout
func.func @fanout(%arg0: f32) -> (f32) attributes {ignore_side_effects_for_testing} {
  %0 = "x.a"(%arg0) : (f32) -> (f32)
  %1 = "y.a"(%0) : (f32) -> (f32)
  %2 = "y.b"(%0) : (f32) -> (f32)
  %3 = "y.c"(%0) : (f32) -> (f32)
  %4 = "y.d"(%0) : (f32) -> (f32)
  %5 = "x.b"(%1, %2, %3, %4) : (f32, f32, f32, f32) -> (f32)
  func.return %5 : f32
}
// CHECK: x.a
// CHECK-NEXT: y.a
// CHECK-NEXT: y.b
// CHECK-NEXT: y.c
// CHECK-NEXT: y.d
// CHECK-NEXT: x.b

// -----

// CHECK-LABEL: @constants
func.func @constants() -> f32 attributes {ignore_side_effects_for_testing} {
  %0 = "a.x"() : () -> f32
  %1 = "b.x"() : () -> f32
  %2 = "c.x"() : () -> f32
  %3 = "d.x"(%0, %1, %2) : (f32, f32, f32) -> f32
  return %3 : f32
}
// CHECK-DAG: a.x
// CHECK-DAG: b.x
// CHECK-DAG: c.x
// CHECK-NEXT: d.x

// -----

// CHECK-LABEL: @tf_and_mhlo
func.func @tf_and_mhlo(%arg0: tensor<32x28x28x1xf32>, %arg1: tensor<!tf_type.resource<tensor<3x3x1x5xf32>>>, %arg2: tensor<!tf_type.resource<tensor<5xf32>>>, %arg3: tensor<!tf_type.resource<tensor<3920x10xf32>>>, %arg4: tensor<!tf_type.resource<tensor<10xf32>>>) -> (tensor<32x10xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<32x10xf32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<32x28x28x5xf32>
  %2 = "tf.ReadVariableOp"(%arg4) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<10xf32>
  %3 = "tf.ReadVariableOp"(%arg2) : (tensor<!tf_type.resource<tensor<5xf32>>>) -> tensor<5xf32>
  %4 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<3x3x1x5xf32>>>) -> tensor<3x3x1x5xf32>
  %5 = "tf.ReadVariableOp"(%arg3) : (tensor<!tf_type.resource<tensor<3920x10xf32>>>) -> tensor<3920x10xf32>
  %6 = mhlo.convolution(%arg0, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x28x28x1xf32>, tensor<3x3x1x5xf32>) -> tensor<32x28x28x5xf32>
  %7 = "mhlo.broadcast_in_dim"(%3) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<5xf32>) -> tensor<32x28x28x5xf32>
  %8 = mhlo.add %6, %7 : tensor<32x28x28x5xf32>
  %9 = mhlo.maximum %8, %1 : tensor<32x28x28x5xf32>
  %10 = "mhlo.reshape"(%9) : (tensor<32x28x28x5xf32>) -> tensor<32x3920xf32>
  %11 = "mhlo.dot"(%10, %5) : (tensor<32x3920xf32>, tensor<3920x10xf32>) -> tensor<32x10xf32>
  %12 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<10xf32>) -> tensor<32x10xf32>
  %13 = mhlo.add %11, %12 : tensor<32x10xf32>
  %14 = mhlo.maximum %13, %0 : tensor<32x10xf32>
  return %14 : tensor<32x10xf32>
}
// CHECK: ReadVariableOp
// CHECK: mhlo.convolution
// CHECK: mhlo.add
// CHECK: mhlo.maximum
// CHECK: mhlo.reshape
// CHECK: mhlo.dot
// CHECK: mhlo.add
// CHECK: mhlo.maximum
// CHECK: return{{.*}}tensor<32x10xf32>

// -----

// CHECK-LABEL: @mhlo_while
func.func private @mhlo_while() {
  // CHECK-NEXT: mhlo.constant
  // CHECK-NEXT: mhlo.constant
  // CHECK-NEXT: mhlo.constant
  %0 = mhlo.constant dense<-1> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  %2 = mhlo.constant dense<20> : tensor<i32>
  // CHECK-NEXT: mhlo.while
  %3, %4, %5 = mhlo.while(%iterArg = %1, %iterArg_0 = %0, %iterArg_1 = %1) : tensor<i32>, tensor<i32>, tensor<i32>
   cond {
    %17 = mhlo.compare  LT, %iterArg_1, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %17 : tensor<i1>
  } do {
    mhlo.return %0, %0, %0 : tensor<i32>, tensor<i32>, tensor<i32>
  }
  return
}

// -----

// CHECK-LABEL: @nested_regions
func.func @nested_regions(%arg0: f32) attributes {ignore_side_effects_for_testing} {
  %0 = "x.a"(%arg0) : (f32) -> f32
  %1 = "y.a"(%arg0) : (f32) -> f32
  %2 = "x.b"(%arg0) : (f32) -> f32
  "nested1"() ({
    %3 = "y.a"(%arg0) : (f32) -> f32
    %4 = "x.b"(%1) : (f32) -> f32
    %5 = "y.b"(%2) : (f32) -> f32
    "nested2"() ({
      %6 = "x.b"(%3) : (f32) -> f32
      %7 = "y.c"(%4) : (f32) -> f32
      %8 = "x.d"(%arg0) : (f32) -> f32
      %9 = "y.e"(%6) : (f32) -> f32
    }) : () -> ()
  }) : () -> ()
}
// CHECK: x.a
// CHECK-NEXT: x.b
// CHECK-NEXT: y.a
// CHECK-NEXT: nested1
// CHECK-NEXT: y.a
// CHECK-NEXT: y.b
// CHECK-NEXT: x.b
// CHECK-NEXT: nested2
// CHECK-NEXT: x.b
// CHECK-NEXT: x.d
// CHECK-NEXT: y.c
// CHECK-NEXT: y.e

// -----

// CHECK-LABEL: interleaved_tf_and_mhlo
func.func private @interleaved_tf_and_mhlo() {
  %m0 = mhlo.constant dense<0> : tensor<i32>
  %t0 = "tf.Const"() { value = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
  %m1 = mhlo.constant dense<1> : tensor<i32>
  %t1 = "tf.Const"() { value = dense<1> : tensor<1xi32> } : () -> tensor<1xi32>
  %m2 = mhlo.constant dense<1> : tensor<i32>
  %t2 = "tf.Const"() { value = dense<1> : tensor<1xi32> } : () -> tensor<1xi32>
  %m3 = mhlo.constant dense<1> : tensor<i32>
  %t3 = "tf.Const"() { value = dense<1> : tensor<1xi32> } : () -> tensor<1xi32>
  // CHECK: mhlo.constant
  // CHECK: mhlo.constant
  // CHECK: mhlo.constant
  // CHECK: mhlo.constant
  // CHECK: tf.Const
  // CHECK: tf.Const
  // CHECK: tf.Const
  // CHECK: tf.Const
  return
}

// -----

// CHECK-LABEL: variable_ops
func.func private @variable_ops(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
  %t3 = "tf.Const"() { value = dense<0> : tensor<0xi32> } : () -> tensor<0xi32>
  // Without side effect analysis, we would now schedule tf.ReadVariableOp next,
  // since all its operands are ready. Check that we don't.
  %0 = mhlo.constant dense<0.> : tensor<f32>
  "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK: tf.Const
  // CHECK: mhlo.constant
  // CHECK: tf.Assign
  // CHECK: tf.Read
  return
}

// -----

func.func private @id(%arg0: tensor<!tf_type.variant>) -> tensor<!tf_type.variant> {
  return %arg0 : tensor<!tf_type.variant>
}

// CHECK-LABEL: iterators
func.func private @iterators(%arg0 : tensor<!tf_type.variant>) {
  %0 = "tf.Iterator"() {container = "", output_shapes = [#tf_type.shape<200x28x28x1>, #tf_type.shape<200x10>], output_types = [f32, f32], shared_name = "_iterator1"} : () -> tensor<!tf_type.resource>
  %1 = func.call @id(%arg0) : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
  "tf.MakeIterator"(%1, %0) {_class = ["loc:@BatchDatasetV2"], device = ""} : (tensor<!tf_type.variant>, tensor<!tf_type.resource>) -> ()
  %2:2 = "tf.IteratorGetNext"(%0) {_class = ["loc:@iterator"], device = ""} : (tensor<!tf_type.resource>) -> (tensor<200x28x28x1xf32>, tensor<200x10xf32>)
  // CHECK: tf.Iterator
  // CHECK: tf.MakeIterator
  // CHECK: tf.IteratorGetNext
  return
}
