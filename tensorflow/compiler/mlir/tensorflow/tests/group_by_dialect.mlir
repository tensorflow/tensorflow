// RUN: tf-opt %s -allow-unregistered-dialect --tf-group-by-dialect --split-input-file | FileCheck %s

func.func @three_dialects(%arg0: tensor<f32>) -> tensor<f32> {
  %one = "glue.constant"() { value = 1: i32 } : () -> i32
  %done = "glue.compare" (%one, %one) { predicate = #glue<"compare LTE"> } : (i32, i32) -> i1
  %2 = mhlo.constant dense<[[1.1]]> : tensor<1x1xf32>
  %3 = mhlo.multiply %2, %2 : tensor<1x1xf32>
  %cst = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.AddV2"(%arg0, %cst) {device = "/device:CPU:0"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<f32>) -> tensor<f32>
  "tf.NoOp"() {device = ""} : () -> ()
  func.return %1 : tensor<f32>
  // CHECK: func @three_dialects
  // CHECK-NEXT: glue.constant
  // CHECK-NEXT: glue.compare
  // CHECK-NEXT: call{{.*}}mhlo
  // CHECK-NEXT: call{{.*}}tf
  // CHECK-NEXT: return

  // CHECK: func.func{{.*}}dialect = "mhlo"
  // CHECK-NEXT: mhlo.constant
  // CHECK-NEXT: mhlo.multiply

  // CHECK: func.func{{.*}}dialect = "tf"
  // CHECK-NEXT: tf.Const
  // CHECK-NEXT: tf.AddV2
  // CHECK-NEXT: tf.Identity
  // CHECK-NEXT: tf.NoOp
  // CHECK-NEXT: return
}

// -----

// Test what happens if we don't preprocess the input function,
// i.e., don't first group operations by dialect.

func.func @interleave(%arg0: f32) -> (f32, f32, f32) {
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

// CHECK: func @interleave
// CHECK-NEXT: call{{.*}}x
// CHECK-NEXT: call{{.*}}y
// CHECK-NEXT: call{{.*}}z
// CHECK-NEXT: call{{.*}}x
// CHECK-NEXT: call{{.*}}y
// CHECK-NEXT: call{{.*}}z
// CHECK-NEXT: call{{.*}}x
// CHECK-NEXT: call{{.*}}y
// CHECK-NEXT: call{{.*}}z
// CHECK-NEXT: return
// CHECK: func{{.*}}dialect = "x"
// CHECK-NEXT: x.a
// CHECK: func{{.*}}dialect = "y"
// CHECK-NEXT: y.a
// CHECK: func{{.*}}dialect = "z"
// CHECK-NEXT: z.a
// CHECK: func{{.*}}dialect = "x"
// CHECK-NEXT: x.b
// CHECK: func{{.*}}dialect = "y"
// CHECK-NEXT: y.b
// CHECK: func{{.*}}dialect = "z"
// CHECK-NEXT: z.b
// CHECK: func{{.*}}dialect = "x"
// CHECK-NEXT: x.c
// CHECK: func{{.*}}dialect = "y"
// CHECK-NEXT: y.c
// CHECK: func{{.*}}dialect = "z"
// CHECK-NEXT: z.c

// -----

func.func @statements(%arg0: f32, %arg1: f32) -> (f32, f32) {
  %0 = "x.a"(%arg0) : (f32) -> f32
  "x.b"(%0) : (f32) -> ()

  %1 = "y.a"(%arg1) : (f32) -> f32
  "y.b"(%1) : (f32) -> ()

  func.return %0, %1 : f32, f32
}

// CHECK: func @statements
// CHECK-NEXT: call{{.*}}x
// CHECK-NEXT: call{{.*}}y
// CHECK-NEXT: return
// CHECK: func{{.*}}dialect = "x"
// CHECK-NEXT: x.a
// CHECK-NEXT: x.b
// CHECK: func{{.*}}dialect = "y"
// CHECK-NEXT: y.a
// CHECK-NEXT: y.b

// -----

func.func @empty(%arg0: i32) -> i32 {
  func.return %arg0 : i32
}

// CHECK: func @empty
// CHECK-NOT: call
// -----

func.func @only_top_level() -> (i32, i32, i32, i32) {
  %0 = "glue.constant"() { value = 0: i32 } : () -> i32
  %1 = "glue.constant"() { value = 1: i32 } : () -> i32
  %2 = "glue.constant"() { value = 2: i32 } : () -> i32
  %3 = "glue.constant"() { value = 3: i32 } : () -> i32
  func.return %0, %1, %2, %3: i32, i32, i32, i32
}

// CHECK: func @only_top_level
// CHECK-NOT: call

// -----

func.func @leaves_processed_funcs_alone() -> (i32, i32)
    attributes {dialect="foo"}
{
  %0 = "foo.constant"() { value = 0: i32 } : () -> i32
  %1 = "foo.constant"() { value = 1: i32 } : () -> i32
  func.return %0, %1: i32, i32
}

// CHECK: func @leaves_processed_funcs_alone
// CHECK-NOT: call

// -----

func.func @handles_mhlo_regions() {
  "glue.someop"() : () -> ()
  %4 = mhlo.constant dense<0.0> : tensor<200x10xf32>
  %13 = mhlo.constant dense<-0.0> : tensor<f32>
  %29 = mhlo.reduce(%4 init: %13) applies mhlo.add across dimensions = [1] : (tensor<200x10xf32>, tensor<f32>) -> tensor<200xf32>
  return
}

// CHECK: func @handles_mhlo_regions
// CHECK: call [[name:@[^(]*]](
// CHECK: func [[name]]()

// -----

func.func @handles_regions_that_use_arguments(%arg0: f32, %arg1: f32) {
  %0 = "glue.someop"() : () -> f32
  "foo.someop"() ({
    "bar.a"(%arg0) : (f32) -> f32
    "bar.b"(%arg1) : (f32) -> f32
    "bar.c"(%0) : (f32) -> f32
  }, {}): () -> ()
  return
}

// CHECK: func @handles_regions_that_use_arguments
// CHECK: call [[foo:@[^(]*]](%arg0, %arg1, %0)
// CHECK: func [[bar:@[^(]*]]
// CHECK-SAME: dialect = "bar"
// CHECK: bar.a
// CHECK: bar.b
// CHECK: bar.c
// CHECK: func [[foo]]
// CHECK: foo.someop
// CHECK: call [[bar]]

// -----

func.func @handles_nested_regions(%arg0: f32, %arg1: f32) {
  %0 = "glue.someop"() : () -> f32
  "foo.someop"() ({
    "foo.some_other_op"() ({
      "bar.a"(%arg0) : (f32) -> f32
      "bar.b"(%arg1) : (f32) -> f32
      "bar.c"(%0) : (f32) -> f32
    }, {}) : () -> ()
  }, {}): () -> ()
  return
}

// CHECK: func @handles_nested_regions
// CHECK: glue.someop
// CHECK: call [[foo:@[^(]*]](%arg0, %arg1, %0)
// CHECK: func [[bar:@[^(]*]](
// CHECK-SAME: dialect = "bar"
// CHECK: bar.a
// CHECK: bar.b
// CHECK: bar.c
// CHECK: func [[foo]]
// CHECK: foo.someop
// CHECK: foo.some_other_op

// -----

func.func @groups_nested_regions(%arg0: f32, %arg1: f32) {
  %0 = "glue.someop"() : () -> f32
  "foo.someop"() ({
    "bar.a"(%arg0) : (f32) -> f32
    "bar.b"(%arg1) : (f32) -> f32
    %1 = "bar.c"(%0) : (f32) -> f32
    "foo.a"(%1) : (f32) -> f32
  }, {}): () -> ()
  return
}

// CHECK: func @groups_nested_regions
// CHECK: func [[bar:@[^(]*]](
// CHECK-SAME: dialect = "bar"
// CHECK-NOT: foo
// CHECK: func
// CHECK-SAME: dialect = "foo"
// CHECK: call [[bar]]
// CHECK: foo.a
