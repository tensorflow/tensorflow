// RUN: tf-opt %s -mlir-print-debuginfo | tf-opt -mlir-print-debuginfo -mlir-print-op-generic | FileCheck %s --dump-input=fail

// This file should be written in the generic form with debug locations.
// (that is, as if printed with `-mlir-print-debuginfo -mlir-print-op-generic`).
// The test parses the file, prints it in the pretty form with debug locations,
// then parses it back, then prints it in the generic form again.
// This should be an exact roundtrip.
// This is a rare example of a test where exact output checking makes
// sense.
//
// To debug this test, it is most useful to look at the output of the first
// tf-opt invocation.



// This test case exercises the "tf_executor.island wraps" syntax.
// When parsing it back, we should recover all 3 locations (the
// tf_executor.island, tf.Identity, and tf_executor.yield).

// CHECK-LABEL: "func"
// CHECK:    "tf_executor.graph"() ( {
// CHECK-NEXT:      "tf_executor.island"() ( {
// CHECK-NEXT:        "tf.Identity"(%{{.*}}) : (tensor<f32>) -> tensor<f32> loc("identity@some_function")
// CHECK-NEXT:        "tf_executor.yield"(%{{.*}}) : (tensor<f32>) -> () loc("identity@some_function")
// CHECK-NEXT:      }) : () -> (tensor<f32>, !tf_executor.control) loc("identity@some_function")
// CHECK-NEXT:      "tf_executor.fetch"(%{{.*}}) : (tensor<f32>) -> () loc(unknown)
// CHECK-NEXT:    }) : () -> tensor<f32> loc(unknown)
// CHECK-NEXT:    "std.return"(%{{.*}}) : (tensor<f32>) -> () loc(unknown)
// CHECK-NEXT: sym_name = "island_one_op_all_locs_same"

func @island_one_op_all_locs_same(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "tf_executor.graph"() ( {
    %1:2 = "tf_executor.island"() ( {
      %2 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32> loc("identity@some_function")
      "tf_executor.yield"(%2) : (tensor<f32>) -> () loc("identity@some_function")
    }) : () -> (tensor<f32>, !tf_executor.control) loc("identity@some_function")
    "tf_executor.fetch"(%1#0) : (tensor<f32>) -> () loc(unknown)
  }) : () -> tensor<f32> loc(unknown)
  "std.return"(%0) : (tensor<f32>) -> () loc(unknown)
} loc(unknown)

// This test cases exercises our handling of the "tf_executor.island wraps"
// syntax. In particular, that syntax only prints out a single location, so
// it is incorrect to use that syntax if the island, wrapped op, and yield
// don't have identical locations.

// CHECK-LABEL: "func"
// CHECK:    "tf_executor.graph"() ( {
// CHECK-NEXT:      "tf_executor.island"() ( {
// CHECK-NEXT:        "tf.Identity"(%{{.*}}) : (tensor<f32>) -> tensor<f32> loc("identity@some_function")
// CHECK-NEXT:        "tf_executor.yield"(%{{.*}}) : (tensor<f32>) -> () loc("identity@some_function")
// CHECK-NEXT:      }) : () -> (tensor<f32>, !tf_executor.control) loc("NOT_identity@some_function")
// CHECK-NEXT:      "tf_executor.fetch"(%{{.*}}) : (tensor<f32>) -> () loc(unknown)
// CHECK-NEXT:    }) : () -> tensor<f32> loc(unknown)
// CHECK-NEXT:    "std.return"(%{{.*}}) : (tensor<f32>) -> () loc(unknown)
// CHECK-NEXT: sym_name = "island_one_op_all_locs_NOT_same"

func @island_one_op_all_locs_NOT_same(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "tf_executor.graph"() ( {
    %1:2 = "tf_executor.island"() ( {
      %2 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32> loc("identity@some_function")
      "tf_executor.yield"(%2) : (tensor<f32>) -> () loc("identity@some_function")
    }) : () -> (tensor<f32>, !tf_executor.control) loc("NOT_identity@some_function")
    "tf_executor.fetch"(%1#0) : (tensor<f32>) -> () loc(unknown)
  }) : () -> tensor<f32> loc(unknown)
  "std.return"(%0) : (tensor<f32>) -> () loc(unknown)
} loc(unknown)
