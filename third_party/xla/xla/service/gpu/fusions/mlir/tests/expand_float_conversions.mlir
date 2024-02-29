// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-expand-conversions="include-bf16=true" -canonicalize | FileCheck %s -check-prefixes=CHECK,CHECK-BF16
// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-expand-conversions="include-bf16=false" -canonicalize | FileCheck %s -check-prefixes=CHECK,CHECK-NO-BF16

module {
  func.func @f64_to_bf16(%arg0: f64) -> bf16 {
    %ret = arith.truncf %arg0 : f64 to bf16
    return %ret : bf16
  }
}

// CHECK-LABEL:    f64_to_bf16
// CHECK-SAME:         (%[[ARG:.*]]: f64)
// CHECK-BF16:       arith.truncf %[[ARG]] : f64 to f32
// CHECK-BF16-NOT:   arith.truncf

// CHECK-NO-BF16:    %[[F32:.*]] = arith.truncf %[[ARG]] : f64 to f32
// CHECK-NO-BF16:    arith.truncf %[[F32]] : f32 to bf16


module {
  func.func @bf16_to_f64(%arg0: bf16) -> f64 {
    %ret = arith.extf %arg0 : bf16 to f64
    return %ret : f64
  }
}

// CHECK-LABEL:    bf16_to_f64
// CHECK:            bitcast {{.*}} : i32 to f32
// CHECK:            arith.extf {{.*}} : f32 to f64

// -----

module {
  func.func @bf16_to_int(%arg0: bf16) -> i32 {
    %ret = arith.fptosi %arg0 : bf16 to i32
    return %ret : i32
  }
}

// CHECK-LABEL: bf16_to_int
// CHECK:         arith.fptosi {{.*}} : f32 to i32

// -----

module {
  func.func @int_to_bf16(%arg0: i16) -> bf16 {
    %ret = arith.sitofp %arg0 : i16 to bf16
    return %ret : bf16
  }
}

// CHECK-LABEL: int_to_bf16
// CHECK:         arith.sitofp {{.*}} : i16 to f32
