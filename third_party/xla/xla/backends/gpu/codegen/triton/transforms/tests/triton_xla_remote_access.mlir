// RUN: xla-opt %s --triton-xla-remote-access | FileCheck %s

// CHECK-LABEL: module {
// CHECK-LABEL: tt.func @get_rank(%arg0: !tt.ptr<i64>) -> i64 {
tt.func @get_rank(
  %metadata: !tt.ptr<i64>
) -> (i64) {
  // CHECK-NOT: triton_xla.get_rank
  // CHECK: %0 = tt.load %arg0 : !tt.ptr<i64>
  %rank = triton_xla.get_rank %metadata : !tt.ptr<i64> -> i64
  tt.return %rank : i64
}
