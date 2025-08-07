// RUN: emitters_opt %s -split-input-file -xla-cpu-add-reduction-fast-math-flags | FileCheck %s

func.func @caller(%x: f32, %y: f32) -> f32
{
  %z = func.call @reducer(%x, %y) { xla.is_reduction }: (f32, f32) -> f32
  func.return %z : f32
}

func.func @reducer(%x: f32, %y: f32) -> f32
{
  %z = arith.addf %x, %y : f32
  func.return %z : f32
}

// CHECK-LABEL: func.func @caller
// CHECK-LABEL: func.func @reducer
// CHECK arith.addf {{.*}} fastmath<reassoc> : f32
