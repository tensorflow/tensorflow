// RUN: emitters_opt %s -xla-expand-integer-power --split-input-file  \
// RUN: | FileCheck %s

func.func @expand_integer_power_scalar(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-NOT: math.ipowi
  %0 = math.ipowi %arg0, %arg1 : i32
  func.return %0 : i32
}

//-----

func.func @expand_integer_power_vector(
    %arg0: vector<4xi32>,
    %arg1: vector<4xi32>) -> vector<4xi32> {
  // CHECK-NOT: math.ipowi
  %0 = math.ipowi %arg0, %arg1 : vector<4xi32>
  func.return %0 : vector<4xi32>
}

//-----

func.func @expand_integer_power_tensor(
    %arg0: tensor<6x4xi32>,
    %arg1: tensor<6x4xi32>) -> tensor<6x4xi32> {
  // CHECK-NOT: math.ipowi
  %0 = math.ipowi %arg0, %arg1 : tensor<6x4xi32>
  func.return %0 : tensor<6x4xi32>
}
