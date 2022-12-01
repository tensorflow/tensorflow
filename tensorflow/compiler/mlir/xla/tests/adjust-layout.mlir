// RUN: xla-opt -pass-pipeline='builtin.module(func.func(xla-adjust-layout))' %s | FILECHECK_OPTS="" FileCheck %s

func.func @infeed_dequeue_tuple() -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>) {
  // CHECK: [[TOKEN:%.*]] = mhlo.create_token : !mhlo.token
  %0 = "mhlo.create_token"() : () -> !mhlo.token

  // CHECK: [[INFEED:%.*]]:3 = "mhlo.infeed"([[TOKEN]]) {infeed_config = "", layout = [{{\[1, 3, 2, 0], \[1, 2, 0]}}]} : (!mhlo.token) -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>, !mhlo.token)
  %1:3 = "mhlo.infeed"(%0) {infeed_config = ""} : (!mhlo.token) -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>, !mhlo.token)

  // CHECK: return [[INFEED]]#0, [[INFEED]]#1
  func.return %1#0, %1#1 : tensor<1x8x4x4xi32>, tensor<1x100x1xf32>
}
