// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func private @compare(%a: f32, %b: f32) -> tensor<16xi1> {
  %false = arith.cmpf false, %a, %b : f32
  %oeq = arith.cmpf oeq, %a, %b : f32
  %ogt = arith.cmpf ogt, %a, %b : f32
  %oge = arith.cmpf oge, %a, %b : f32

  %olt = arith.cmpf olt, %a, %b : f32
  %ole = arith.cmpf ole, %a, %b : f32
  %one = arith.cmpf one, %a, %b : f32
  %ord = arith.cmpf ord, %a, %b : f32

  %ueq = arith.cmpf ueq, %a, %b : f32
  %ugt = arith.cmpf ugt, %a, %b : f32
  %uge = arith.cmpf uge, %a, %b : f32
  %ult = arith.cmpf ult, %a, %b : f32

  %ule = arith.cmpf ule, %a, %b : f32
  %une = arith.cmpf une, %a, %b : f32
  %uno = arith.cmpf uno, %a, %b : f32
  %true = arith.cmpf true, %a, %b : f32

  %ret = tensor.from_elements
    %false, %oeq, %ogt, %oge,
    %olt, %ole, %one, %ord,
    %ueq, %ugt, %uge, %ult,
    %ule, %une, %uno, %true : tensor<16xi1>
  func.return %ret : tensor<16xi1>
}

func.func @nan_vs_one() -> tensor<16xi1> {
  %nan = arith.constant 0x7fc00000 : f32
  %one = arith.constant 1.0 : f32

  %ret = func.call @compare(%nan, %one) : (f32, f32) -> tensor<16xi1>
  func.return %ret : tensor<16xi1>
}

// CHECK-LABEL: @nan_vs_one
// CHECK-NEXT: Results
// CHECK-NEXT: [false, false, false, false,
// CHECK-SAME:  false, false, false, false,
// CHECK-SAME:  true, true, true, true,
// CHECK-SAME:  true, true, true, true]

func.func @one_vs_nan() -> tensor<16xi1> {
  %nan = arith.constant 0x7fc00000 : f32
  %one = arith.constant 1.0 : f32

  %ret = func.call @compare(%one, %nan) : (f32, f32) -> tensor<16xi1>
  func.return %ret : tensor<16xi1>
}

// CHECK-LABEL: @one_vs_nan
// CHECK-NEXT: Results
// CHECK-NEXT: [false, false, false, false,
// CHECK-SAME:  false, false, false, false,
// CHECK-SAME:  true, true, true, true,
// CHECK-SAME:  true, true, true, true]

func.func @nan_vs_nan() -> tensor<16xi1> {
  %nan = arith.constant 0x7fc00000 : f32

  %ret = func.call @compare(%nan, %nan) : (f32, f32) -> tensor<16xi1>
  func.return %ret : tensor<16xi1>
}

// CHECK-LABEL: @nan_vs_nan
// CHECK-NEXT: Results
// CHECK-NEXT: [false, false, false, false,
// CHECK-SAME:  false, false, false, false,
// CHECK-SAME:  true, true, true, true,
// CHECK-SAME:  true, true, true, true]

func.func @one_vs_one() -> tensor<16xi1> {
  %one = arith.constant 1.0 : f32

  %ret = func.call @compare(%one, %one) : (f32, f32) -> tensor<16xi1>
  func.return %ret : tensor<16xi1>
}

// CHECK-LABEL: @one_vs_one
// CHECK-NEXT: Results
// CHECK-NEXT: [false, true, false, true,
// CHECK-SAME:  false, true, false, true,
// CHECK-SAME:  true, false, true, false,
// CHECK-SAME:  true, false, false, true]

func.func @one_vs_two() -> tensor<16xi1> {
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32

  %ret = func.call @compare(%one, %two) : (f32, f32) -> tensor<16xi1>
  func.return %ret : tensor<16xi1>
}

// CHECK-LABEL: @one_vs_two
// CHECK-NEXT: Results
// CHECK-NEXT: [false, false, false, false,
// CHECK-SAME:  true, true, true, true,
// CHECK-SAME:  false, false, false, true,
// CHECK-SAME:  true, true, false, true]

func.func @two_vs_one() -> tensor<16xi1> {
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32

  %ret = func.call @compare(%two, %one) : (f32, f32) -> tensor<16xi1>
  func.return %ret : tensor<16xi1>
}

// CHECK-LABEL: @two_vs_one
// CHECK-NEXT: Results
// CHECK-NEXT: [false, false, true, true,
// CHECK-SAME:  false, false, true, true,
// CHECK-SAME:  false, true, true, false,
// CHECK-SAME:  false, true, false, true]

func.func @vector() -> vector<4xi1> {
  %0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
  %1 = arith.constant dense<[10.0, 2.0, 30.0, 4.0]> : vector<4xf32>
  %ret = arith.cmpf oeq, %0, %1 : vector<4xf32>
  return %ret : vector<4xi1>
}

// CHECK-LABEL: @vector
// CHECK-NEXT: Results
// CHECK-NEXT: vector<4xi1>: [false, true, false, true]
