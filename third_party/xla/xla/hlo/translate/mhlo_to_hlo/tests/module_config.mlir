// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s

// CHECK: HloModule check_imported_configs, {{.*}} replica_count=2, num_partitions=4
module @check_imported_configs attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 2 : i32} {
  func.func @main(%arg0: tensor<1xf32>) -> (tensor<1xf32>) {
    return %arg0 : tensor<1xf32>
  }
}