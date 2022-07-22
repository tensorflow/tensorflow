// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main.{{.*}} () -> u64[2]
// CHECK-NEXT: ROOT %rng-get-and-update-state.1 = u64[2] rng-get-and-update-state(), delta=1
func.func @main() -> tensor<2xui64> {
  %1 = mhlo.xla.rng_get_and_update_state {delta = 1: i64}
  func.return %1 : tensor<2xui64>
}
