// RUN: emitters_opt %s -split-input-file -xla-cpu-expand-float-ops | FileCheck %s

func.func @trunc(%input: f32) -> bf16 {
  %truncated = arith.truncf %input : f32 to bf16
  func.return %truncated : bf16
}
// CHECK-NOT: arith.truncf

// -----


func.func @extend(%input: bf16) -> f32 {
  %truncated = arith.extf %input : bf16 to f32
  func.return %truncated : f32
}

// CHECK-NOT: arith.extf
