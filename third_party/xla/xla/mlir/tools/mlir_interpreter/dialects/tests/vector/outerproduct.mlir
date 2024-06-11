// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @outerproduct_1d_1d() -> vector<2x3xi32> {
  %a = arith.constant dense<[1, 2]> : vector<2xi32>
  %b = arith.constant dense<[3, 4, 5]> : vector<3xi32>
  %o = vector.outerproduct %a, %b : vector<2xi32>, vector<3xi32>
  return %o : vector<2x3xi32>
}

// CHECK-LABEL: @outerproduct_1d_1d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[3, 4, 5], [6, 8, 10]]

func.func @outerproduct_1d_1d_add() -> vector<2x3xi32> {
  %a = arith.constant dense<[1, 2]> : vector<2xi32>
  %b = arith.constant dense<[3, 4, 5]> : vector<3xi32>
  %init = arith.constant dense<[[100, 200, 300], [400, 500, 600]]>
    : vector<2x3xi32>
  %o = vector.outerproduct %a, %b, %init : vector<2xi32>, vector<3xi32>
  return %o : vector<2x3xi32>
}

// CHECK-LABEL: @outerproduct_1d_1d_add
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[103, 204, 305], [406, 508, 610]]

func.func @outerproduct_1d_0d_and() -> vector<1xi32> {
  %a = arith.constant dense<[3]> : vector<1xi32>
  %b = arith.constant 1 : i32
  %init = arith.constant dense<[9]> : vector<1xi32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<and>}
    : vector<1xi32>, i32
  return %o : vector<1xi32>
}

// CHECK-LABEL: @outerproduct_1d_0d_and
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [1]

func.func @outerproduct_1d_0d_maxui() -> vector<3xi32> {
  %a = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %b = arith.constant 3 : i32
  %init = arith.constant dense<[100, 0, -100]> : vector<3xi32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<maxui>}
    : vector<3xi32>, i32
  return %o : vector<3xi32>
}

// CHECK-LABEL: @outerproduct_1d_0d_maxui
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [100, 6, -100]

func.func @outerproduct_1d_0d_mul() -> vector<3xi32> {
  %a = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %b = arith.constant 3 : i32
  %init = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<mul>}
    : vector<3xi32>, i32
  return %o : vector<3xi32>
}

// CHECK-LABEL: @outerproduct_1d_0d_mul
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [3, 12, 27]

func.func @outerproduct_1d_0d_minui() -> vector<3xi32> {
  %a = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %b = arith.constant 3 : i32
  %init = arith.constant dense<[100, 0, -100]> : vector<3xi32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<minui>}
    : vector<3xi32>, i32
  return %o : vector<3xi32>
}

// CHECK-LABEL: @outerproduct_1d_0d_minui
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [3, 0, 9]

func.func @outerproduct_1d_0d_maxsi() -> vector<3xi32> {
  %a = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %b = arith.constant 3 : i32
  %init = arith.constant dense<[100, 0, -100]> : vector<3xi32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<maxsi>}
    : vector<3xi32>, i32
  return %o : vector<3xi32>
}

// CHECK-LABEL: @outerproduct_1d_0d_maxsi
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [100, 6, 9]

func.func @outerproduct_1d_0d_minsi() -> vector<3xi32> {
  %a = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %b = arith.constant 3 : i32
  %init = arith.constant dense<[100, 0, -100]> : vector<3xi32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<minsi>}
    : vector<3xi32>, i32
  return %o : vector<3xi32>
}

// CHECK-LABEL: @outerproduct_1d_0d_minsi
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [3, 0, -100]

func.func @outerproduct_1d_0d_maxf() -> vector<1xf32> {
  %a = arith.constant dense<[1.0]> : vector<1xf32>
  %b = arith.constant 3.0 : f32
  %init = arith.constant dense<[10.0]> : vector<1xf32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<maximumf>}
    : vector<1xf32>, f32
  return %o : vector<1xf32>
}

// CHECK-LABEL: @outerproduct_1d_0d_maxf
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [1.000000e+01]

func.func @outerproduct_1d_0d_minf() -> vector<1xf32> {
  %a = arith.constant dense<[1.0]> : vector<1xf32>
  %b = arith.constant 3.0 : f32
  %init = arith.constant dense<[10.0]> : vector<1xf32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<minimumf>}
    : vector<1xf32>, f32
  return %o : vector<1xf32>
}

// CHECK-LABEL: @outerproduct_1d_0d_minf
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [3.000000e+00]

func.func @outerproduct_1d_0d_or() -> vector<1xi32> {
  %a = arith.constant dense<[3]> : vector<1xi32>
  %b = arith.constant 1 : i32
  %init = arith.constant dense<[9]> : vector<1xi32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<or>}
    : vector<1xi32>, i32
  return %o : vector<1xi32>
}

// CHECK-LABEL: @outerproduct_1d_0d_or
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [11]

func.func @outerproduct_1d_0d_xor() -> vector<1xi32> {
  %a = arith.constant dense<[3]> : vector<1xi32>
  %b = arith.constant 1 : i32
  %init = arith.constant dense<[9]> : vector<1xi32>
  %o = vector.outerproduct %a, %b, %init {kind = #vector.kind<xor>}
    : vector<1xi32>, i32
  return %o : vector<1xi32>
}

// CHECK-LABEL: @outerproduct_1d_0d_xor
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [10]
