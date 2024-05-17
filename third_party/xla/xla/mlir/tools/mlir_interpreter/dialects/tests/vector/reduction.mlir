// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @reduction_add() -> i32 {
  %a = arith.constant dense<[10, 4, 7]> : vector<3xi32>
  %r = vector.reduction <add>, %a : vector<3xi32> into i32
  return %r : i32
}

// CHECK-LABEL: @reduction_add
// CHECK-NEXT: Results
// CHECK-NEXT: 21

func.func @reduction_minf_degenerate() -> f32 {
  %a = arith.constant dense<[1.0]> : vector<1xf32>
  %r = vector.reduction <minimumf>, %a : vector<1xf32> into f32
  return %r : f32
}

// CHECK-LABEL: @reduction_minf_degenerate
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00

func.func @reduction_minf() -> f32 {
  %a = arith.constant dense<[10.0, 4.0, 7.0]> : vector<3xf32>
  %r = vector.reduction <minimumf>, %a : vector<3xf32> into f32
  return %r : f32
}

// CHECK-LABEL: @reduction_minf
// CHECK-NEXT: Results
// CHECK-NEXT: 4.000000e+00

func.func @reduction_acc() -> f32 {
  %a = arith.constant dense<[10.0, 4.0, 7.0]> : vector<3xf32>
  %acc = arith.constant 3.0 : f32
  %r = vector.reduction <minimumf>, %a, %acc : vector<3xf32> into f32
  return %r : f32
}

// CHECK-LABEL: @reduction_acc
// CHECK-NEXT: Results
// CHECK-NEXT: 3.000000e+00

func.func @reduction_acc_first_is_minimum() -> f32 {
  %a = arith.constant dense<[1.0, 4.0, 7.0]> : vector<3xf32>
  %acc = arith.constant 3.0 : f32
  %r = vector.reduction <minimumf>, %a, %acc : vector<3xf32> into f32
  return %r : f32
}

// CHECK-LABEL: @reduction_acc_first_is_minimum
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00

func.func @masked_and() -> i32 {
  %a = arith.constant dense<[255, 127, 6]> : vector<3xi32>
  %m = arith.constant dense<[true, true, false]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <and>, %a : vector<3xi32> into i32
  } : vector<3xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @masked_and
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 127

func.func @masked_xor() -> i32 {
  %a = arith.constant dense<[255, 1, 3]> : vector<3xi32>
  %m = arith.constant dense<[false, true, true]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <xor>, %a : vector<3xi32> into i32
  } : vector<3xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @masked_xor
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 2

func.func @masked_or() -> i32 {
  %a = arith.constant dense<[255, 1, 3]> : vector<3xi32>
  %m = arith.constant dense<[false, true, true]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <or>, %a : vector<3xi32> into i32
  } : vector<3xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @masked_or
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 3

func.func @masked_add_i32() -> i32 {
  %a = arith.constant dense<[255, 1, 3]> : vector<3xi32>
  %m = arith.constant dense<[false, true, true]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <add>, %a : vector<3xi32> into i32
  } : vector<3xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @masked_add_i32
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 4

func.func @masked_add_f32() -> f32 {
  %a = arith.constant dense<[255.0, 1.0, 3.0]> : vector<3xf32>
  %m = arith.constant dense<[false, true, true]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <add>, %a : vector<3xf32> into f32
  } : vector<3xi1> -> f32
  return %r : f32
}

// CHECK-LABEL: @masked_add_f32
// CHECK-NEXT: Results
// CHECK-NEXT: f32: 4.0

func.func @masked_mul_i32() -> i32 {
  %a = arith.constant dense<[255, 2, 3]> : vector<3xi32>
  %m = arith.constant dense<[false, true, true]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <mul>, %a : vector<3xi32> into i32
  } : vector<3xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @masked_mul_i32
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 6

func.func @masked_mul_f32() -> f32 {
  %a = arith.constant dense<[255.0, 2.0, 3.0]> : vector<3xf32>
  %m = arith.constant dense<[false, true, true]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <mul>, %a : vector<3xf32> into f32
  } : vector<3xi1> -> f32
  return %r : f32
}

// CHECK-LABEL: @masked_mul_f32
// CHECK-NEXT: Results
// CHECK-NEXT: f32: 6.0

func.func @masked_minsi() -> i32 {
  %a = arith.constant dense<[255, -2, -5]> : vector<3xi32>
  %m = arith.constant dense<[true, true, false]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <minsi>, %a : vector<3xi32> into i32
  } : vector<3xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @masked_minsi
// CHECK-NEXT: Results
// CHECK-NEXT: i32: -2

func.func @masked_minui() -> i32 {
  %a = arith.constant dense<[255, -2, -5]> : vector<3xi32>
  %m = arith.constant dense<[true, true, false]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <minui>, %a : vector<3xi32> into i32
  } : vector<3xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @masked_minui
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 255

func.func @masked_maxsi() -> i32 {
  %a = arith.constant dense<[255, -2, 500]> : vector<3xi32>
  %m = arith.constant dense<[true, true, false]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <maxsi>, %a : vector<3xi32> into i32
  } : vector<3xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @masked_maxsi
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 255

func.func @masked_maxui() -> i32 {
  %a = arith.constant dense<[255, -5, -2]> : vector<3xi32>
  %m = arith.constant dense<[true, true, false]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <maxui>, %a : vector<3xi32> into i32
  } : vector<3xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @masked_maxui
// CHECK-NEXT: Results
// CHECK-NEXT: i32: -5

func.func @masked_minf() -> f32 {
  %a = arith.constant dense<[255.0, 2.0, 3.0]> : vector<3xf32>
  %m = arith.constant dense<[true, false, true]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <minimumf>, %a : vector<3xf32> into f32
  } : vector<3xi1> -> f32
  return %r : f32
}

// CHECK-LABEL: @masked_minf
// CHECK-NEXT: Results
// CHECK-NEXT: f32: 3.0

func.func @masked_maxf() -> f32 {
  %a = arith.constant dense<[255.0, 2.0, 3.0]> : vector<3xf32>
  %m = arith.constant dense<[false, true, true]> : vector<3xi1>
  %r = vector.mask %m {
    vector.reduction <maximumf>, %a : vector<3xf32> into f32
  } : vector<3xi1> -> f32
  return %r : f32
}

// CHECK-LABEL: @masked_maxf
// CHECK-NEXT: Results
// CHECK-NEXT: f32: 3.0

func.func @masked_maxf_empty() -> f32 {
  %a = arith.constant dense<[255.0]> : vector<1xf32>
  %m = arith.constant dense<[false]> : vector<1xi1>
  %r = vector.mask %m {
    vector.reduction <maximumf>, %a : vector<1xf32> into f32
  } : vector<1xi1> -> f32
  return %r : f32
}

// CHECK-LABEL: @masked_maxf_empty
// CHECK-NEXT: Results
// CHECK-NEXT: f32: -INF
