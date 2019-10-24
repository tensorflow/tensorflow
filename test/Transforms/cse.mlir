// RUN: mlir-opt %s -pass-pipeline='func(cse)' | FileCheck %s

// CHECK-DAG: #map0 = (d0) -> (d0 mod 2)
#map0 = (d0) -> (d0 mod 2)

// CHECK-LABEL: @simple_constant
func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %c1_i32 = constant 1 : i32
  %0 = constant 1 : i32

  // CHECK-NEXT: return %c1_i32, %c1_i32 : i32, i32
  %1 = constant 1 : i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL: @basic
func @basic() -> (index, index) {
  // CHECK: %c0 = constant 0 : index
  %c0 = constant 0 : index
  %c1 = constant 0 : index

  // CHECK-NEXT: %0 = affine.apply #map0(%c0)
  %0 = affine.apply #map0(%c0)
  %1 = affine.apply #map0(%c1)

  // CHECK-NEXT: return %0, %0 : index, index
  return %0, %1 : index, index
}

// CHECK-LABEL: @many
func @many(f32, f32) -> (f32) {
^bb0(%a : f32, %b : f32):
  // CHECK-NEXT: %0 = addf %arg0, %arg1 : f32
  %c = addf %a, %b : f32
  %d = addf %a, %b : f32
  %e = addf %a, %b : f32
  %f = addf %a, %b : f32

  // CHECK-NEXT: %1 = addf %0, %0 : f32
  %g = addf %c, %d : f32
  %h = addf %e, %f : f32
  %i = addf %c, %e : f32

  // CHECK-NEXT: %2 = addf %1, %1 : f32
  %j = addf %g, %h : f32
  %k = addf %h, %i : f32

  // CHECK-NEXT: %3 = addf %2, %2 : f32
  %l = addf %j, %k : f32

  // CHECK-NEXT: return %3 : f32
  return %l : f32
}

/// Check that operations are not eliminated if they have different operands.
// CHECK-LABEL: @different_ops
func @different_ops() -> (i32, i32) {
  // CHECK: %c0_i32 = constant 0 : i32
  // CHECK: %c1_i32 = constant 1 : i32
  %0 = constant 0 : i32
  %1 = constant 1 : i32

  // CHECK-NEXT: return %c0_i32, %c1_i32 : i32, i32
  return %0, %1 : i32, i32
}

/// Check that operations are not eliminated if they have different result
/// types.
// CHECK-LABEL: @different_results
func @different_results(%arg0: tensor<*xf32>) -> (tensor<?x?xf32>, tensor<4x?xf32>) {
  // CHECK: %0 = tensor_cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  // CHECK-NEXT: %1 = tensor_cast %arg0 : tensor<*xf32> to tensor<4x?xf32>
  %0 = tensor_cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  %1 = tensor_cast %arg0 : tensor<*xf32> to tensor<4x?xf32>

  // CHECK-NEXT: return %0, %1 : tensor<?x?xf32>, tensor<4x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<4x?xf32>
}

/// Check that operations are not eliminated if they have different attributes.
// CHECK-LABEL: @different_attributes
func @different_attributes(index, index) -> (i1, i1, i1) {
^bb0(%a : index, %b : index):
  // CHECK: %0 = cmpi "slt", %arg0, %arg1 : index
  %0 = cmpi "slt", %a, %b : index

  // CHECK-NEXT: %1 = cmpi "ne", %arg0, %arg1 : index
  /// Predicate 1 means inequality comparison.
  %1 = cmpi "ne", %a, %b : index
  %2 = "std.cmpi"(%a, %b) {predicate = 1} : (index, index) -> i1

  // CHECK-NEXT: return %0, %1, %1 : i1, i1, i1
  return %0, %1, %2 : i1, i1, i1
}

/// Check that operations with side effects are not eliminated.
// CHECK-LABEL: @side_effect
func @side_effect() -> (memref<2x1xf32>, memref<2x1xf32>) {
  // CHECK: %0 = alloc() : memref<2x1xf32>
  %0 = alloc() : memref<2x1xf32>

  // CHECK-NEXT: %1 = alloc() : memref<2x1xf32>
  %1 = alloc() : memref<2x1xf32>

  // CHECK-NEXT: return %0, %1 : memref<2x1xf32>, memref<2x1xf32>
  return %0, %1 : memref<2x1xf32>, memref<2x1xf32>
}

/// Check that operation definitions are properly propagated down the dominance
/// tree.
// CHECK-LABEL: @down_propagate_for
func @down_propagate_for() {
  // CHECK: %c1_i32 = constant 1 : i32
  %0 = constant 1 : i32

  // CHECK-NEXT: affine.for {{.*}} = 0 to 4 {
  affine.for %i = 0 to 4 {
    // CHECK-NEXT: "foo"(%c1_i32, %c1_i32) : (i32, i32) -> ()
    %1 = constant 1 : i32
    "foo"(%0, %1) : (i32, i32) -> ()
  }
  return
}

// CHECK-LABEL: @down_propagate
func @down_propagate() -> i32 {
  // CHECK-NEXT: %c1_i32 = constant 1 : i32
  %0 = constant 1 : i32

  // CHECK-NEXT: %true = constant 1 : i1
  %cond = constant 1 : i1

  // CHECK-NEXT: cond_br %true, ^bb1, ^bb2(%c1_i32 : i32)
  cond_br %cond, ^bb1, ^bb2(%0 : i32)

^bb1: // CHECK: ^bb1:
  // CHECK-NEXT: br ^bb2(%c1_i32 : i32)
  %1 = constant 1 : i32
  br ^bb2(%1 : i32)

^bb2(%arg : i32):
  return %arg : i32
}

/// Check that operation definitions are NOT propagated up the dominance tree.
// CHECK-LABEL: @up_propagate_for
func @up_propagate_for() -> i32 {
  // CHECK: affine.for {{.*}} = 0 to 4 {
  affine.for %i = 0 to 4 {
    // CHECK-NEXT: %c1_i32_0 = constant 1 : i32
    // CHECK-NEXT: "foo"(%c1_i32_0) : (i32) -> ()
    %0 = constant 1 : i32
    "foo"(%0) : (i32) -> ()
  }

  // CHECK: %c1_i32 = constant 1 : i32
  // CHECK-NEXT: return %c1_i32 : i32
  %1 = constant 1 : i32
  return %1 : i32
}

// CHECK-LABEL: func @up_propagate
func @up_propagate() -> i32 {
  // CHECK-NEXT:  %c0_i32 = constant 0 : i32
  %0 = constant 0 : i32

  // CHECK-NEXT: %true = constant 1 : i1
  %cond = constant 1 : i1

  // CHECK-NEXT: cond_br %true, ^bb1, ^bb2(%c0_i32 : i32)
  cond_br %cond, ^bb1, ^bb2(%0 : i32)

^bb1: // CHECK: ^bb1:
  // CHECK-NEXT: %c1_i32 = constant 1 : i32
  %1 = constant 1 : i32

  // CHECK-NEXT: br ^bb2(%c1_i32 : i32)
  br ^bb2(%1 : i32)

^bb2(%arg : i32): // CHECK: ^bb2
  // CHECK-NEXT: %c1_i32_0 = constant 1 : i32
  %2 = constant 1 : i32

  // CHECK-NEXT: %1 = addi %0, %c1_i32_0 : i32
  %add = addi %arg, %2 : i32

  // CHECK-NEXT: return %1 : i32
  return %add : i32
}

/// The same test as above except that we are testing on a cfg embedded within
/// an operation region.
// CHECK-LABEL: func @up_propagate_region
func @up_propagate_region() -> i32 {
  // CHECK-NEXT: %0 = "foo.region"
  %0 = "foo.region"() ({
    // CHECK-NEXT:  %c0_i32 = constant 0 : i32
    // CHECK-NEXT: %true = constant 1 : i1
    // CHECK-NEXT: cond_br

    %1 = constant 0 : i32
    %true = constant 1 : i1
    cond_br %true, ^bb1, ^bb2(%1 : i32)

  ^bb1: // CHECK: ^bb1:
    // CHECK-NEXT: %c1_i32 = constant 1 : i32
    // CHECK-NEXT: br

    %c1_i32 = constant 1 : i32
    br ^bb2(%c1_i32 : i32)

  ^bb2(%arg : i32): // CHECK: ^bb2(%1: i32):
    // CHECK-NEXT: %c1_i32_0 = constant 1 : i32
    // CHECK-NEXT: %2 = addi %1, %c1_i32_0 : i32
    // CHECK-NEXT: "foo.yield"(%2) : (i32) -> ()

    %c1_i32_0 = constant 1 : i32
    %2 = addi %arg, %c1_i32_0 : i32
    "foo.yield" (%2) : (i32) -> ()
  }) : () -> (i32)
  return %0 : i32
}

/// This test checks that nested regions that are isolated from above are
/// properly handled.
// CHECK-LABEL: @nested_isolated
func @nested_isolated() -> i32 {
  // CHECK-NEXT: constant 1
  %0 = constant 1 : i32

  // CHECK-NEXT: @nested_func
  func @nested_func() {
    // CHECK-NEXT: constant 1
    %foo = constant 1 : i32
    "foo.yield"(%foo) : (i32) -> ()
  }

  // CHECK: "foo.region"
  "foo.region"() ({
    // CHECK-NEXT: constant 1
    %foo = constant 1 : i32
    "foo.yield"(%foo) : (i32) -> ()
  }) : () -> ()

  return %0 : i32
}
