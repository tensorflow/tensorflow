// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main() -> tensor<1x10xf32> {
  %result = "xla_hlo.iota"() {
    iota_dimension = 1 : i64
  } : () -> tensor<1x10xf32>
  return %result : tensor<1x10xf32>
}

// CHECK-LABEL:main
// CHECK:  ROOT %[[RESULT:.*]] = f32[1,10] iota(), iota_dimension=1
