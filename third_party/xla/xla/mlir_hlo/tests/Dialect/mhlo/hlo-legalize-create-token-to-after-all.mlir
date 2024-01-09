// RUN: mlir-hlo-opt -mhlo-legalize-create-token-to-after-all -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @create_token_to_after_all
func.func @create_token_to_after_all() -> !mhlo.token {
  // CHECK: [[RES:%.+]] = mhlo.after_all : !mhlo.token
  %0 = "mhlo.create_token"() : () -> !mhlo.token
  func.return %0 : !mhlo.token
}
