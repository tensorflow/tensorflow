// RUN: xla-opt -pass-pipeline='builtin.func(xla-adjust-layout)' %s | FILECHECK_OPTS="" FileCheck %s

func @infeed_dequeue_tuple() -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>) {
  // CHECK: [[TOKEN:%.*]] = "mhlo.create_token"() : () -> !mhlo.token
  %0 = "mhlo.create_token"() : () -> !mhlo.token
  // CHECK: [[INFEED:%.*]] = "mhlo.infeed"([[TOKEN]]) {infeed_config = "", layout = [{{\[\[1, 3, 2, 0], \[1, 2, 0]]}}, unit]} : (!mhlo.token) -> tuple<tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>, !mhlo.token>
  %1 = "mhlo.infeed"(%0) {infeed_config = ""} : (!mhlo.token) -> tuple<tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>, !mhlo.token>
  // CHECK: [[INFEED_VAL:%.*]] = "mhlo.get_tuple_element"([[INFEED]]) {index = 0 : i32} : (tuple<tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>, !mhlo.token>) -> tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>
  %2 = "mhlo.get_tuple_element"(%1) {index = 0 : i32} : (tuple<tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>, !mhlo.token>) -> tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>
  // CHECK: [[RES_1:%.*]] = "mhlo.get_tuple_element"([[INFEED_VAL]]) {index = 0 : i32} : (tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>) -> tensor<1x8x4x4xi32>
  %3 = "mhlo.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>) -> tensor<1x8x4x4xi32>
  // CHECK: [[RES_2:%.*]] = "mhlo.get_tuple_element"([[INFEED_VAL]]) {index = 1 : i32} : (tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>) -> tensor<1x100x1xf32>
  %4 = "mhlo.get_tuple_element"(%2) {index = 1 : i32} : (tuple<tensor<1x8x4x4xi32>, tensor<1x100x1xf32>>) -> tensor<1x100x1xf32>
  // CHECK: return [[RES_1]], [[RES_2]]
  return %3, %4 : tensor<1x8x4x4xi32>, tensor<1x100x1xf32>
}
