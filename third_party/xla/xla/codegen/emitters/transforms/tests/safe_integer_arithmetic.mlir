// RUN: emitters_opt %s -xla-safe-integer-arithmetic --split-input-file  \
// RUN: | FileCheck %s

func.func @unmarked_signed_div_is_not_changed(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[RESULT:.*]] = arith.divsi %arg0, %arg1 : i32
  %0 = arith.divsi %arg0, %arg1 : i32
  // CHECK: return %[[RESULT]] : i32
  func.return %0 : i32
}

// -----

func.func @unmarked_unsigned_div_is_not_changed(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[RESULT:.*]] = arith.divui %arg0, %arg1 : i32
  %0 = arith.divui %arg0, %arg1 : i32
  // CHECK: return %[[RESULT]] : i32
  func.return %0 : i32
}

// -----

func.func @unmarked_signed_rem_is_not_changed(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[RESULT:.*]] = arith.remsi %arg0, %arg1 : i32
  %0 = arith.remsi %arg0, %arg1 : i32
  // CHECK: return %[[RESULT]] : i32
  func.return %0 : i32
}

// -----

func.func @unmarked_unsigned_rem_is_not_changed(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[RESULT:.*]] = arith.remui %arg0, %arg1 : i32
  %0 = arith.remui %arg0, %arg1 : i32
  // CHECK: return %[[RESULT]] : i32
  func.return %0 : i32
}

// -----

func.func @signed_div(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-DAG: %[[C_1:.*]] = arith.constant -1 : i32
  // CHECK-DAG: %[[CMIN:.*]] = arith.constant -2147483648 : i32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:     %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : i32
  // CHECK:     %[[LHS_MIN:.*]] = arith.cmpi eq, %arg0, %[[CMIN]] : i32
  // CHECK:     %[[RHS_MINUS1:.*]] = arith.cmpi eq, %arg1, %[[C_1]] : i32
  // CHECK:     %[[OVERFLOW:.*]] = arith.andi %[[LHS_MIN]], %[[RHS_MINUS1]] : i1
  // CHECK:     %[[IS_UB:.*]] = arith.ori %[[RHS_ZERO]], %[[OVERFLOW]] : i1
  // CHECK:     %[[BOUNDED_RHS:.*]] = arith.select %[[IS_UB]], %[[C1]], %arg1 : i32
  // CHECK:     %[[DIV:.*]] = arith.divsi %arg0, %[[BOUNDED_RHS]] : i32
  // CHECK:     %[[OVERFLOW_RESULT:.*]] = arith.select %[[OVERFLOW]], %[[CMIN]], %[[DIV]] : i32
  // CHECK:     %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %[[C_1]], %[[OVERFLOW_RESULT]] : i32
  %0 = arith.divsi %arg0, %arg1 {xla.guard_ub} : i32
  // CHECK: return %[[RESULT]] : i32
  func.return %0 : i32
}

// -----

func.func @signed_div_vector(%arg0: vector<8xi32>, %arg1: vector<8xi32>) -> vector<8xi32> {
  // CHECK-DAG: %[[C_1:.*]] = arith.constant dense<-1> : vector<8xi32>
  // CHECK-DAG: %[[CMIN:.*]] = arith.constant dense<-2147483648> : vector<8xi32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0> : vector<8xi32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant dense<1> : vector<8xi32>
  // CHECK:     %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : vector<8xi32>
  // CHECK:     %[[LHS_MIN:.*]] = arith.cmpi eq, %arg0, %[[CMIN]] : vector<8xi32>
  // CHECK:     %[[RHS_MINUS1:.*]] = arith.cmpi eq, %arg1, %[[C_1]] : vector<8xi32>
  // CHECK:     %[[OVERFLOW:.*]] = arith.andi %[[LHS_MIN]], %[[RHS_MINUS1]] : vector<8xi1>
  // CHECK:     %[[IS_UB:.*]] = arith.ori %[[RHS_ZERO]], %[[OVERFLOW]] : vector<8xi1>
  // CHECK:     %[[BOUNDED_RHS:.*]] = arith.select %[[IS_UB]], %[[C1]], %arg1 : vector<8xi1>, vector<8xi32>
  // CHECK:     %[[DIV:.*]] = arith.divsi %arg0, %[[BOUNDED_RHS]] : vector<8xi32>
  // CHECK:     %[[OVERFLOW_RESULT:.*]] = arith.select %[[OVERFLOW]], %[[CMIN]], %[[DIV]] : vector<8xi1>, vector<8xi32>
  // CHECK:     %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %[[C_1]], %[[OVERFLOW_RESULT]] : vector<8xi1>, vector<8xi32>
  %0 = arith.divsi %arg0, %arg1 {xla.guard_ub} : vector<8xi32>
  // CHECK: return %[[RESULT]] : vector<8xi32>
  func.return %0 : vector<8xi32>
}

// -----

func.func @signed_div_tensor(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  // CHECK-DAG: %[[C_1:.*]] = arith.constant dense<-1> : tensor<8xi32>
  // CHECK-DAG: %[[CMIN:.*]] = arith.constant dense<-2147483648> : tensor<8xi32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0> : tensor<8xi32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant dense<1> : tensor<8xi32>
  // CHECK:     %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : tensor<8xi32>
  // CHECK:     %[[LHS_MIN:.*]] = arith.cmpi eq, %arg0, %[[CMIN]] : tensor<8xi32>
  // CHECK:     %[[RHS_MINUS1:.*]] = arith.cmpi eq, %arg1, %[[C_1]] : tensor<8xi32>
  // CHECK:     %[[OVERFLOW:.*]] = arith.andi %[[LHS_MIN]], %[[RHS_MINUS1]] : tensor<8xi1>
  // CHECK:     %[[IS_UB:.*]] = arith.ori %[[RHS_ZERO]], %[[OVERFLOW]] : tensor<8xi1>
  // CHECK:     %[[BOUNDED_RHS:.*]] = arith.select %[[IS_UB]], %[[C1]], %arg1 : tensor<8xi1>, tensor<8xi32>
  // CHECK:     %[[DIV:.*]] = arith.divsi %arg0, %[[BOUNDED_RHS]] : tensor<8xi32>
  // CHECK:     %[[OVERFLOW_RESULT:.*]] = arith.select %[[OVERFLOW]], %[[CMIN]], %[[DIV]] : tensor<8xi1>, tensor<8xi32>
  // CHECK:     %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %[[C_1]], %[[OVERFLOW_RESULT]] : tensor<8xi1>, tensor<8xi32>
  %0 = arith.divsi %arg0, %arg1 {xla.guard_ub} : tensor<8xi32>
  // CHECK: return %[[RESULT]] : tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// -----


func.func @unsigned_div(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-DAG: %[[C_1:.*]] = arith.constant -1 : i32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : i32
  // CHECK: %[[BOUNDED_RHS:.*]] = arith.select %[[RHS_ZERO]], %[[C1]], %arg1 : i32
  // CHECK: %[[DIV:.*]] = arith.divui %arg0, %[[BOUNDED_RHS]] : i32
  // CHECK: %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %[[C_1]], %[[DIV]] : i32
  %0 = arith.divui %arg0, %arg1 {xla.guard_ub} : i32
  // CHECK: return %[[RESULT]] : i32
  func.return %0 : i32
}

// -----

func.func @unsigned_div_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi32> {
  // CHECK-DAG: %[[C_1:.*]] = arith.constant dense<-1> : vector<4xi32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0> : vector<4xi32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant dense<1> : vector<4xi32>
  // CHECK: %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : vector<4xi32>
  // CHECK: %[[BOUNDED_RHS:.*]] = arith.select %[[RHS_ZERO]], %[[C1]], %arg1 :  vector<4xi1>, vector<4xi32>
  // CHECK: %[[DIV:.*]] = arith.divui %arg0, %[[BOUNDED_RHS]] : vector<4xi32>
  // CHECK: %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %[[C_1]], %[[DIV]] :  vector<4xi1>, vector<4xi32>
  %0 = arith.divui %arg0, %arg1 {xla.guard_ub} : vector<4xi32>
  // CHECK: return %[[RESULT]] : vector<4xi32>
  func.return %0 : vector<4xi32>
}

// -----

func.func @unsigned_div_tensor(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-DAG: %[[C_1:.*]] = arith.constant dense<-1> : tensor<4xi32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0> : tensor<4xi32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant dense<1> : tensor<4xi32>
  // CHECK: %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : tensor<4xi32>
  // CHECK: %[[BOUNDED_RHS:.*]] = arith.select %[[RHS_ZERO]], %[[C1]], %arg1 :  tensor<4xi1>, tensor<4xi32>
  // CHECK: %[[DIV:.*]] = arith.divui %arg0, %[[BOUNDED_RHS]] : tensor<4xi32>
  // CHECK: %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %[[C_1]], %[[DIV]] :  tensor<4xi1>, tensor<4xi32>
  %0 = arith.divui %arg0, %arg1 {xla.guard_ub} : tensor<4xi32>
  // CHECK: return %[[RESULT]] : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

func.func @signed_rem(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-DAG: %[[C_1:.*]] = arith.constant -1 : i32
  // CHECK-DAG: %[[CMIN:.*]] = arith.constant -2147483648 : i32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK:     %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : i32
  // CHECK:     %[[LHS_MIN:.*]] = arith.cmpi eq, %arg0, %[[CMIN]] : i32
  // CHECK:     %[[RHS_MINUS1:.*]] = arith.cmpi eq, %arg1, %[[C_1]] : i32
  // CHECK:     %[[OVERFLOW:.*]] = arith.andi %[[LHS_MIN]], %[[RHS_MINUS1]] : i1
  // CHECK:     %[[IS_UB:.*]] = arith.ori %[[RHS_ZERO]], %[[OVERFLOW]] : i1
  // CHECK:     %[[BOUNDED_RHS:.*]] = arith.select %[[IS_UB]], %[[C1]], %arg1 : i32
  // CHECK:     %[[REM:.*]] = arith.remsi %arg0, %[[BOUNDED_RHS]] : i32
  // CHECK:     %[[OVERFLOW_RESULT:.*]] = arith.select %[[OVERFLOW]], %[[C0]], %[[REM]] : i32
  // CHECK:     %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %arg0, %[[OVERFLOW_RESULT]] : i32
  %0 = arith.remsi %arg0, %arg1 {xla.guard_ub} : i32
  // CHECK: return %[[RESULT]] : i32
  func.return %0 : i32
}

// -----

func.func @signed_rem_vector(%arg0: vector<2xi32>, %arg1: vector<2xi32>) -> vector<2xi32> {
  // CHECK-DAG: %[[C_1:.*]] = arith.constant dense<-1> : vector<2xi32>
  // CHECK-DAG: %[[CMIN:.*]] = arith.constant dense<-2147483648> : vector<2xi32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0> : vector<2xi32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant dense<1> : vector<2xi32>
  // CHECK:     %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : vector<2xi32>
  // CHECK:     %[[LHS_MIN:.*]] = arith.cmpi eq, %arg0, %[[CMIN]] : vector<2xi32>
  // CHECK:     %[[RHS_MINUS1:.*]] = arith.cmpi eq, %arg1, %[[C_1]] : vector<2xi32>
  // CHECK:     %[[OVERFLOW:.*]] = arith.andi %[[LHS_MIN]], %[[RHS_MINUS1]] : vector<2xi1>
  // CHECK:     %[[IS_UB:.*]] = arith.ori %[[RHS_ZERO]], %[[OVERFLOW]] : vector<2xi1>
  // CHECK:     %[[BOUNDED_RHS:.*]] = arith.select %[[IS_UB]], %[[C1]], %arg1 : vector<2xi1>, vector<2xi32>
  // CHECK:     %[[REM:.*]] = arith.remsi %arg0, %[[BOUNDED_RHS]] : vector<2xi32>
  // CHECK:     %[[OVERFLOW_RESULT:.*]] = arith.select %[[OVERFLOW]], %[[C0]], %[[REM]] : vector<2xi1>, vector<2xi32>
  // CHECK:     %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %arg0, %[[OVERFLOW_RESULT]] : vector<2xi1>, vector<2xi32>
  %0 = arith.remsi %arg0, %arg1 {xla.guard_ub} : vector<2xi32>
  // CHECK: return %[[RESULT]] : vector<2xi32>
  func.return %0 : vector<2xi32>
}

// -----

func.func @signed_rem_tensor(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-DAG: %[[C_1:.*]] = arith.constant dense<-1> : tensor<2xi32>
  // CHECK-DAG: %[[CMIN:.*]] = arith.constant dense<-2147483648> : tensor<2xi32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0> : tensor<2xi32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant dense<1> : tensor<2xi32>
  // CHECK:     %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : tensor<2xi32>
  // CHECK:     %[[LHS_MIN:.*]] = arith.cmpi eq, %arg0, %[[CMIN]] : tensor<2xi32>
  // CHECK:     %[[RHS_MINUS1:.*]] = arith.cmpi eq, %arg1, %[[C_1]] : tensor<2xi32>
  // CHECK:     %[[OVERFLOW:.*]] = arith.andi %[[LHS_MIN]], %[[RHS_MINUS1]] : tensor<2xi1>
  // CHECK:     %[[IS_UB:.*]] = arith.ori %[[RHS_ZERO]], %[[OVERFLOW]] : tensor<2xi1>
  // CHECK:     %[[BOUNDED_RHS:.*]] = arith.select %[[IS_UB]], %[[C1]], %arg1 : tensor<2xi1>, tensor<2xi32>
  // CHECK:     %[[REM:.*]] = arith.remsi %arg0, %[[BOUNDED_RHS]] : tensor<2xi32>
  // CHECK:     %[[OVERFLOW_RESULT:.*]] = arith.select %[[OVERFLOW]], %[[C0]], %[[REM]] : tensor<2xi1>, tensor<2xi32>
  // CHECK:     %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %arg0, %[[OVERFLOW_RESULT]] : tensor<2xi1>, tensor<2xi32>
  %0 = arith.remsi %arg0, %arg1 {xla.guard_ub} : tensor<2xi32>
  // CHECK: return %[[RESULT]] : tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @unsigned_rem(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : i32
  // CHECK: %[[BOUNDED_RHS:.*]] = arith.select %[[RHS_ZERO]], %[[C1]], %arg1 : i32
  // CHECK: %[[REM:.*]] = arith.remui %arg0, %[[BOUNDED_RHS]] : i32
  // CHECK: %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %arg0, %[[REM]] : i32
  %0 = arith.remui %arg0, %arg1 {xla.guard_ub} : i32
  // CHECK: return %[[RESULT]] : i32
  func.return %0 : i32
}

// -----

func.func @unsigned_rem_vector(%arg0: vector<16xi32>, %arg1: vector<16xi32>) -> vector<16xi32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0> : vector<16xi32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant dense<1> : vector<16xi32>
  // CHECK: %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : vector<16xi32>
  // CHECK: %[[BOUNDED_RHS:.*]] = arith.select %[[RHS_ZERO]], %[[C1]], %arg1 : vector<16xi1>, vector<16xi32>
  // CHECK: %[[REM:.*]] = arith.remui %arg0, %[[BOUNDED_RHS]] : vector<16xi32>
  // CHECK: %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %arg0, %[[REM]] : vector<16xi1>, vector<16xi32>
  %0 = arith.remui %arg0, %arg1 {xla.guard_ub} : vector<16xi32>
  // CHECK: return %[[RESULT]] : vector<16xi32>
  func.return %0 : vector<16xi32>
}

// -----

func.func @unsigned_rem_tensor(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> tensor<16xi32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant dense<0> : tensor<16xi32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant dense<1> : tensor<16xi32>
  // CHECK: %[[RHS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[C0]] : tensor<16xi32>
  // CHECK: %[[BOUNDED_RHS:.*]] = arith.select %[[RHS_ZERO]], %[[C1]], %arg1 : tensor<16xi1>, tensor<16xi32>
  // CHECK: %[[REM:.*]] = arith.remui %arg0, %[[BOUNDED_RHS]] : tensor<16xi32>
  // CHECK: %[[RESULT:.*]] = arith.select %[[RHS_ZERO]], %arg0, %[[REM]] : tensor<16xi1>, tensor<16xi32>
  %0 = arith.remui %arg0, %arg1 {xla.guard_ub} : tensor<16xi32>
  // CHECK: return %[[RESULT]] : tensor<16xi32>
  func.return %0 : tensor<16xi32>
}
