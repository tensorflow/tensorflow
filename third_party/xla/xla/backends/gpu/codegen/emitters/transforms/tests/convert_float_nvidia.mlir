// RUN: emitters_opt %s -split-input-file -xla-gpu-convert-float-nvidia -canonicalize | FileCheck %s

module {
  func.func @intr_f16_to_f8(%arg0: f16) -> (f8E4M3FN, f8E5M2) {
    %a = arith.truncf %arg0 : f16 to f8E4M3FN
    %b = arith.truncf %arg0 : f16 to f8E5M2
    return %a, %b : f8E4M3FN, f8E5M2
  }
}

// CHECK-LABEL: @intr_f16_to_f8
// CHECK: llvm.nvvm.f16x2.to.e4m3x2.rn
// CHECK: llvm.nvvm.f16x2.to.e5m2x2.rn

// -----

module {
  func.func @intr_bf16_to_f8(%arg0: bf16) -> (f8E4M3FN, f8E5M2) {
    %a = arith.truncf %arg0 : bf16 to f8E4M3FN
    %b = arith.truncf %arg0 : bf16 to f8E5M2
    return %a, %b : f8E4M3FN, f8E5M2
  }
}

// CHECK-LABEL: @intr_bf16_to_f8
// CHECK: arith.extf %{{.+}} : bf16 to f32
// CHECK: llvm.nvvm.ff.to.e4m3x2.rn
// CHECK: llvm.nvvm.ff.to.e5m2x2.rn

// -----

module {
  func.func @intr_f32_to_f8(%arg0: f32) -> (f8E4M3FN, f8E5M2) {
    %a = arith.truncf %arg0 : f32 to f8E4M3FN
    %b = arith.truncf %arg0 : f32 to f8E5M2
    return %a, %b : f8E4M3FN, f8E5M2
  }
}

// CHECK-LABEL: @intr_f32_to_f8
// CHECK: llvm.nvvm.ff.to.e4m3x2.rn
// CHECK: llvm.nvvm.ff.to.e5m2x2.rn

// -----

module {
  func.func @intr_f64_to_f8(%arg0: f64) -> (f8E4M3FN, f8E5M2) {
    %a = arith.truncf %arg0 : f64 to f8E4M3FN
    %b = arith.truncf %arg0 : f64 to f8E5M2
    return %a, %b : f8E4M3FN, f8E5M2
  }
}

// CHECK-LABEL: @intr_f64_to_f8
// CHECK: arith.truncf %{{.+}} : f64 to f32
// CHECK: llvm.nvvm.ff.to.e4m3x2.rn
// CHECK: llvm.nvvm.ff.to.e5m2x2.rn

// -----

module {
  func.func @intr_f8_to_f16(%arg0: f8E4M3FN, %arg1: f8E5M2) -> (f16, f16) {
    %a = arith.extf %arg0 : f8E4M3FN to f16
    %b = arith.extf %arg1 : f8E5M2 to f16
    return %a, %b : f16, f16
  }
}

// CHECK-LABEL: @intr_f8_to_f16
// CHECK: llvm.nvvm.e4m3x2.to.f16x2.rn
// CHECK: llvm.nvvm.e5m2x2.to.f16x2.rn

// -----

module {
  func.func @intr_f8_to_bf16(%arg0: f8E4M3FN, %arg1: f8E5M2) -> (bf16, bf16) {
    %a = arith.extf %arg0 : f8E4M3FN to bf16
    %b = arith.extf %arg1 : f8E5M2 to bf16
    return %a, %b : bf16, bf16
  }
}

// CHECK-LABEL: @intr_f8_to_bf16
// CHECK: llvm.nvvm.e4m3x2.to.f16x2.rn
// CHECK: llvm.nvvm.e5m2x2.to.f16x2.rn
// CHECK: arith.extf %{{.+}} : f16 to f32
// CHECK: arith.truncf %{{.+}} : f32 to bf16

// -----

module {
  func.func @intr_f8_to_f32(%arg0: f8E4M3FN, %arg1: f8E5M2) -> (f32, f32) {
    %a = arith.extf %arg0 : f8E4M3FN to f32
    %b = arith.extf %arg1 : f8E5M2 to f32
    return %a, %b : f32, f32
  }
}

// CHECK-LABEL: @intr_f8_to_f32
// CHECK: llvm.nvvm.e4m3x2.to.f16x2.rn
// CHECK: llvm.nvvm.e5m2x2.to.f16x2.rn
// CHECK: arith.extf %{{.+}} : f16 to f32

// -----

module {
  func.func @intr_f8_to_f8(%arg0: f8E4M3FN) -> f8E5M2 {
    %tmp = arith.extf %arg0 : f8E4M3FN to f16
    %res = arith.truncf %tmp : f16 to f8E5M2
    return %res : f8E5M2
  }
}

// CHECK-LABEL: @intr_f8_to_f8
// CHECK: llvm.nvvm.e4m3x2.to.f16x2.rn
// CHECK: llvm.nvvm.f16x2.to.e5m2x2.rn

// -----

module {
  func.func @intr_f16_to_f8_fix_infinity(%arg0: f16) -> f8E5M2 {
    %res = arith.truncf %arg0 : f16 to f8E5M2
    return %res : f8E5M2
  }
}

// CHECK-LABEL: @intr_f16_to_f8_fix_infinity
// CHECK: %[[PAIR:.*]] = llvm.call_intrinsic "llvm.nvvm.f16x2.to.e5m2x2.rn"
// CHECK: %[[RES:.*]] = llvm.trunc %[[PAIR]] : i16 to i8
// CHECK: %[[INT:.*]] = arith.bitcast %arg0 : f16 to i16
// CHECK: %[[VAL:.*]] = arith.andi %[[INT]], %c32767_i16
// CHECK: %[[LOWER:.*]] = arith.cmpi ugt, %[[VAL]], %c31615_i16
// CHECK: %[[UPPER:.*]] = arith.cmpi ule, %[[VAL]], %c31744_i16
// CHECK: %[[ISINF:.*]] = arith.andi %[[LOWER]], %[[UPPER]]
// CHECK: arith.select %[[ISINF]], {{.*}}, %[[RES]]

// -----

module {
  func.func @intr_f32_to_f8_fix_infinity(%arg0: f32) -> f8E4M3FN {
    %res = arith.truncf %arg0 : f32 to f8E4M3FN
    return %res : f8E4M3FN
  }
}

// CHECK-LABEL: @intr_f32_to_f8_fix_infinity
// CHECK: %[[PAIR:.*]] = llvm.call_intrinsic "llvm.nvvm.ff.to.e4m3x2.rn"
// CHECK: %[[RES:.*]] = llvm.trunc %[[PAIR]] : i16 to i8
// CHECK: %[[INT:.*]] = arith.bitcast %arg0 : f32 to i32
// CHECK: %[[VAL:.*]] = arith.andi %[[INT]], %c2147483647_i32
// CHECK: %[[LOWER:.*]] = arith.cmpi ugt, %[[VAL]], %c1139277824_i32
// CHECK: %[[UPPER:.*]] = arith.cmpi ule, %[[VAL]], %c2139095040_i32
// CHECK: %[[ISINF:.*]] = arith.andi %[[LOWER]], %[[UPPER]]
// CHECK: arith.select %[[ISINF]], {{.*}}, %[[RES]]
