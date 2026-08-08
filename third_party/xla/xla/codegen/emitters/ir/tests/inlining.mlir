// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: emitters_opt %s -split-input-file -xla-erase-dead-functions -inline | FileCheck %s

module {
  func.func private @mul(%a: f32, %b: f32) -> f32 {
    %ret = arith.mulf %a, %b : f32
    return %ret : f32
  }

  func.func private @add(%a: f32, %b: f32) -> f32 {
    %add = arith.addf %a, %b : f32
    %ret = xla.pure_call @mul(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %ret = xla.pure_call @add(%a, %b) : (f32, f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @caller
// CHECK-NOT: xla.pure_call @add
// CHECK: arith.addf
// CHECK-NOT: xla.pure_call @mul
// CHECK: arith.mulf

// -----

module {
  func.func private @mul(%a: f32, %b: f32) -> f32 {
    %ret = arith.mulf %a, %b : f32
    return %ret : f32
  }

  func.func private @add(%a: f32, %b: f32) -> f32 {
    %add = arith.addf %a, %b : f32
    %ret = xla.pure_call @mul(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %ret = xla.pure_call @add(%a, %b) {noinline} : (f32, f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK:         func.func {{.*}}@add
// CHECK:           arith.addf
// CHECK-NOT:       xla.pure_call @mul
// CHECK:           arith.mulf
// CHECK:         func.func {{.*}}@caller
// CHECK:           xla.pure_call @add

// -----

module {
  func.func @fused_computation(%arg0: tensor<2xf32> {xla.slice_index = 0 : index}, %arg1: tensor<2xf32> {xla.slice_index = 1 : index}, %arg2: tensor<2xf32> {xla.slice_index = 2 : index}) -> tensor<2xf32> attributes {xla.entry} {
    %0 = gpu.thread_id  x {xla.range = [0 : index, 1 : index]}
    %1 = xla.pure_call @fused_computation_atan2(%arg0, %arg1, %0) : (tensor<2xf32>, tensor<2xf32>, index) -> f32
    %inserted = tensor.insert %1 into %arg2[%0] : tensor<2xf32>
    return %inserted : tensor<2xf32>
  }
  func.func private @fused_computation_atan2(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: index {xla.range = [0 : index, 1 : index]}) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[%arg2] : tensor<2xf32>
    %extracted_0 = tensor.extract %arg1[%arg2] : tensor<2xf32>
    %0 = arith.addf %extracted, %extracted_0 : f32
    %1 = arith.subf %extracted, %extracted_0 : f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.divf %0, %1 : f32
    %4 = math.atan2 %2, %3 : f32
    return %4 : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @fused_computation
// CHECK-NOT: xla.pure_call @add
// CHECK: gpu.thread_id
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.subf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.divf
// CHECK-NEXT: math.atan2
// CHECK-NEXT: tensor.insert

// -----

module {
  // Do not inline this function as it has two callers. Even if the callers are
  // in different functions at the start, after inlining the two callers are in
  // the same function.
  func.func private @large(%a: f32, %b: f32) -> f32 {
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %a, %mul : f32
    %div = arith.divf %add, %b : f32
    %sub = arith.subf %div, %a : f32
    %atan2 = math.atan2 %b, %sub : f32
    %neg = arith.negf %atan2 : f32
    %zero = arith.constant 0.0 : f32
    %comp = arith.cmpf olt, %neg, %zero : f32
    %ret = arith.select %comp, %zero, %neg : f32
    return %ret : f32
  }

  func.func private @add(%a: f32, %b: f32) -> f32 {
    %add = arith.addf %a, %b : f32
    %ret = xla.pure_call @large(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %add = xla.pure_call @add(%a, %b) : (f32, f32) -> (f32)
    %ret = xla.pure_call @large(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @caller
// CHECK: arith.addf
// CHECK: xla.pure_call @large
// CHECK: xla.pure_call @large

// -----

module {
  // On CPU, small multi-caller functions are inlined.
  func.func private @large(%a: f32, %b: f32) -> f32 {
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %a, %mul : f32
    %div = arith.divf %add, %b : f32
    %sub = arith.subf %div, %a : f32
    %atan2 = math.atan2 %b, %sub : f32
    %neg = arith.negf %atan2 : f32
    %zero = arith.constant 0.0 : f32
    %comp = arith.cmpf olt, %neg, %zero : f32
    %ret = arith.select %comp, %zero, %neg : f32
    return %ret : f32
  }

  func.func private @add(%a: f32, %b: f32) -> f32 {
    %add = arith.addf %a, %b : f32
    %ret = xla.pure_call @large(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32
      attributes { xla.backend_kind = #xla.backend_kind<cpu> } {
    %add = xla.pure_call @add(%a, %b) : (f32, f32) -> (f32)
    %ret = xla.pure_call @large(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @caller
// CHECK-NOT: xla.pure_call
// CHECK: math.atan2
// CHECK-NOT: xla.pure_call
// CHECK: math.atan2
// CHECK-NOT: xla.pure_call

// -----

module {
  func.func private @add(%a: f32, %b: f32) -> f32 {
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %add = xla.pure_call @add(%a, %b) : (f32, f32) -> (f32)
    %ret = xla.pure_call @add(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @caller
// CHECK-COUNT-2: xla.pure_call

// -----

module {
  func.func private @fib0(%start : f32) -> f32 attributes {no_compute = true} {
    %zero = arith.constant 0.0 : f32
    return %zero : f32
  }
  func.func private @fib1(%start : f32) -> f32 attributes {no_compute = true} {
    return %start : f32
  }
  func.func private @fib2(%start : f32) -> f32 {
    %a = xla.pure_call @fib0(%start) : (f32) -> (f32)
    %b = xla.pure_call @fib1(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @fib3(%start : f32) -> f32 {
    %a = xla.pure_call @fib1(%start) : (f32) -> (f32)
    %b = xla.pure_call @fib2(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @fib4(%start : f32) -> f32 {
    %a = xla.pure_call @fib2(%start) : (f32) -> (f32)
    %b = xla.pure_call @fib3(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  // When inlining the other functions into @fib5, this function exceeds the
  // threshold for inlining.
  func.func private @fib5(%start : f32) -> f32 {
    %a = xla.pure_call @fib3(%start) : (f32) -> (f32)
    %b = xla.pure_call @fib4(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  // As we do not inline @fib5 into @fib6, this function stays below the
  // threshold for inlining.
  func.func private @fib6(%start : f32) -> f32 {
    %a = xla.pure_call @fib4(%start) : (f32) -> (f32)
    %b = xla.pure_call @fib5(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @fib7(%start : f32) -> f32 {
    %a = xla.pure_call @fib5(%start) : (f32) -> (f32)
    %b = xla.pure_call @fib6(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }

  func.func @caller(%a: f32) -> f32
      attributes { xla.backend_kind = #xla.backend_kind<cpu> } {
    %ret = xla.pure_call @fib7(%a) : (f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK-NOT: fib0
// CHECK-NOT: fib1
// CHECK-NOT: fib2
// CHECK-NOT: fib3
// CHECK-NOT: fib4
// CHECK-NOT: fib5
// CHECK-NOT: fib6
// CHECK-NOT: fib7

// CHECK: @caller
// CHECK-NOT: xla.pure_call

// -----

module {
  func.func private @complex(%a: f32, %b: f32) -> complex<f32> {
    %ret = complex.create %a, %b : complex<f32>
    return %ret : complex<f32>
  }

  func.func @caller(%a: f32, %b: f32) -> complex<f32> {
    %ret = xla.pure_call @complex(%a, %b) : (f32, f32) -> (complex<f32>)
    return %ret : complex<f32>
  }
}

// CHECK-LABEL: module {
// CHECK: @caller
// CHECK-NEXT: complex.create

// -----

module {
  func.func private @callee2(%a: f32) -> f32 {
    %ret = arith.addf %a, %a : f32
    return %ret : f32
  }

  func.func private @callee1(%a: f32) -> f32 {
    %c1 = xla.pure_call @callee2(%a) : (f32) -> (f32)
    %b0 = arith.addf %a, %a : f32
    %b1 = arith.addf %b0, %a : f32
    %b2 = arith.addf %b1, %a : f32
    %b3 = arith.addf %b2, %a : f32
    %b4 = arith.addf %b3, %a : f32
    %b5 = arith.addf %b4, %a : f32
    %b6 = arith.addf %b5, %a : f32
    %b7 = arith.addf %b6, %a : f32
    %c2 = xla.pure_call @callee2(%b7) : (f32) -> (f32)
    %ret = arith.addf %c1, %c2 : f32
    return %ret : f32
  }

  func.func private @dead(%a: f32) -> f32 {
    %ret = xla.pure_call @callee1(%a) : (f32) -> (f32)
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %ret = xla.pure_call @callee1(%a) : (f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK:      func.func private @callee2
// CHECK-NOT:  func.func private @callee1
// CHECK:      func.func @caller
// CHECK:        pure_call @callee2
// CHECK-NOT: func.func

// -----

// Do not inline multi-caller functions if the callers share no callees.
module {
  func.func private @callee1(%a: f32) -> f32 {
    %b0 = arith.addf %a, %a : f32
    %b1 = arith.addf %b0, %a : f32
    %b2 = arith.addf %b1, %a : f32
    %b3 = arith.addf %b2, %a : f32
    %b4 = arith.addf %b3, %a : f32
    %b5 = arith.addf %b4, %a : f32
    %b6 = arith.addf %b5, %a : f32
    %b7 = arith.addf %b6, %a : f32
    %b8 = arith.addf %b7, %a : f32
    %b9 = arith.addf %b8, %a : f32
    %b10 = arith.addf %b9, %a : f32
    %b11 = arith.addf %b10, %a : f32
    return %b11 : f32
  }

  func.func private @callee2(%a: f32) -> f32 {
    %call = xla.pure_call @callee1(%a) : (f32) -> (f32)
    %b0 = arith.addf %a, %a : f32
    %b1 = arith.addf %b0, %a : f32
    %b2 = arith.addf %b1, %a : f32
    %b3 = arith.addf %b2, %a : f32
    %b4 = arith.addf %b3, %a : f32
    %b5 = arith.addf %b4, %a : f32
    %b6 = arith.addf %b5, %a : f32
    %b7 = arith.addf %b6, %a : f32
    %b8 = arith.addf %b7, %a : f32
    %b9 = arith.addf %b8, %a : f32
    %ret = arith.addf %call, %b9 : f32
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %call1 = xla.pure_call @callee2(%a) : (f32) -> (f32)
    %call2 = xla.pure_call @callee1(%a) : (f32) -> (f32)
    %ret = arith.addf %call1, %call2 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK:         func.func private @callee1
// CHECK-NOT:     callee2
// CHECK:         func.func @caller
// CHECK-COUNT-2: pure_call @callee1

// -----

module {
  func.func private @callee1(%a: f32) -> f32 {
    %b0 = arith.addf %a, %a : f32
    %b1 = arith.addf %b0, %a : f32
    %b2 = arith.addf %b1, %a : f32
    %b3 = arith.addf %b2, %a : f32
    %b4 = arith.addf %b3, %a : f32
    %b5 = arith.addf %b4, %a : f32
    %b6 = arith.addf %b5, %a : f32
    %b7 = arith.addf %b6, %a : f32
    %b8 = arith.addf %b7, %a : f32
    %b9 = arith.addf %b8, %a : f32
    %b10 = arith.addf %b9, %a : f32
    %b11 = arith.addf %b10, %a : f32
    return %b11 : f32
  }

  func.func private @callee2(%a: f32) -> f32 {
    %call = xla.pure_call @callee1(%a) : (f32) -> (f32)
    %b0 = arith.addf %a, %a : f32
    %b1 = arith.addf %b0, %a : f32
    %b2 = arith.addf %b1, %a : f32
    %b3 = arith.addf %b2, %a : f32
    %b4 = arith.addf %b3, %a : f32
    %b5 = arith.addf %b4, %a : f32
    %b6 = arith.addf %b5, %a : f32
    %b7 = arith.addf %b6, %a : f32
    %b8 = arith.addf %b7, %a : f32
    %b9 = arith.addf %b8, %a : f32
    %ret = arith.addf %call, %b9 : f32
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32
      attributes { xla.backend_kind = #xla.backend_kind<cpu> } {
    %call1 = xla.pure_call @callee2(%a) : (f32) -> (f32)
    %call2 = xla.pure_call @callee1(%a) : (f32) -> (f32)
    %ret = arith.addf %call1, %call2 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK:     func.func @caller
// CHECK-NOT: pure_call

// -----

module {
  func.func private @has_no_compute(%a: f32) -> f32
      attributes {no_compute = true} {
    return %a : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %call1 = xla.pure_call @has_no_compute(%a) : (f32) -> (f32)
    %call2 = xla.pure_call @has_no_compute(%b) : (f32) -> (f32)
    %sum = arith.addf %call1, %call2 : f32
    return %sum : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @caller
// CHECK-NEXT: arith.addf
// CHECK-NEXT: return

// -----

// Chain where consecutive levels share no callees (e.g. rotation/quaternion
// chains): every component of one level calls every component of the previous
// level exactly once. All functions are small, so they must all be inlined;
// leaving call-per-use in place compounds recomputation exponentially with
// chain depth.
module {
  func.func private @x0(%a: f32, %b: f32) -> f32 {
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @y0(%a: f32, %b: f32) -> f32 {
    %ret = arith.mulf %a, %b : f32
    return %ret : f32
  }
  func.func private @x1(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x0(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y0(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.subf %x, %y : f32
    return %ret : f32
  }
  func.func private @y1(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x0(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y0(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.divf %x, %y : f32
    return %ret : f32
  }
  func.func @caller(%a: f32, %b: f32) -> f32
      attributes { xla.backend_kind = #xla.backend_kind<cpu> } {
    %x = xla.pure_call @x1(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y1(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.addf %x, %y : f32
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @caller
// CHECK-NOT: xla.pure_call

// -----

// The same no-overlap chain on a GPU module: the aggressive multi-caller
// policy is CPU-only, so the shared leaf functions keep their calls.
module {
  func.func private @x0(%a: f32, %b: f32) -> f32 {
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @y0(%a: f32, %b: f32) -> f32 {
    %ret = arith.mulf %a, %b : f32
    return %ret : f32
  }
  func.func private @x1(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x0(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y0(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.subf %x, %y : f32
    return %ret : f32
  }
  func.func private @y1(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x0(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y0(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.divf %x, %y : f32
    return %ret : f32
  }
  func.func @caller_gpu(%a: f32, %b: f32) -> f32
      attributes { xla.backend_kind = #xla.backend_kind<gpu> } {
    %x = xla.pure_call @x1(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y1(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.addf %x, %y : f32
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @caller_gpu
// CHECK: xla.pure_call @x0
// CHECK: xla.pure_call @y0

// -----

// The same no-overlap chain with no backend attribute: backends that do not
// stamp a kind (for example TPU) keep the old policy.
module {
  func.func private @x0(%a: f32, %b: f32) -> f32 {
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @y0(%a: f32, %b: f32) -> f32 {
    %ret = arith.mulf %a, %b : f32
    return %ret : f32
  }
  func.func private @x1(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x0(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y0(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.subf %x, %y : f32
    return %ret : f32
  }
  func.func private @y1(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x0(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y0(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.divf %x, %y : f32
    return %ret : f32
  }
  func.func @caller_unattributed(%a: f32, %b: f32) -> f32 {
    %x = xla.pure_call @x1(%a, %b) : (f32, f32) -> (f32)
    %y = xla.pure_call @y1(%a, %b) : (f32, f32) -> (f32)
    %ret = arith.addf %x, %y : f32
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @caller_unattributed
// CHECK: xla.pure_call @x0
// CHECK: xla.pure_call @y0

// -----

// CPU multi-caller inlining is capped by callee size: a callee above the
// threshold keeps its calls even on a CPU module.
module {
  func.func private @big(%a: f32) -> f32 {
    %b0 = arith.addf %a, %a : f32
    %b1 = arith.addf %b0, %a : f32
    %b2 = arith.addf %b1, %a : f32
    %b3 = arith.addf %b2, %a : f32
    %b4 = arith.addf %b3, %a : f32
    %b5 = arith.addf %b4, %a : f32
    %b6 = arith.addf %b5, %a : f32
    %b7 = arith.addf %b6, %a : f32
    %b8 = arith.addf %b7, %a : f32
    %b9 = arith.addf %b8, %a : f32
    %b10 = arith.addf %b9, %a : f32
    %b11 = arith.addf %b10, %a : f32
    %b12 = arith.addf %b11, %a : f32
    %b13 = arith.addf %b12, %a : f32
    %b14 = arith.addf %b13, %a : f32
    %b15 = arith.addf %b14, %a : f32
    %b16 = arith.addf %b15, %a : f32
    %b17 = arith.addf %b16, %a : f32
    %b18 = arith.addf %b17, %a : f32
    %b19 = arith.addf %b18, %a : f32
    %b20 = arith.addf %b19, %a : f32
    %b21 = arith.addf %b20, %a : f32
    %b22 = arith.addf %b21, %a : f32
    %b23 = arith.addf %b22, %a : f32
    %b24 = arith.addf %b23, %a : f32
    %b25 = arith.addf %b24, %a : f32
    %b26 = arith.addf %b25, %a : f32
    %b27 = arith.addf %b26, %a : f32
    %b28 = arith.addf %b27, %a : f32
    %b29 = arith.addf %b28, %a : f32
    %b30 = arith.addf %b29, %a : f32
    %b31 = arith.addf %b30, %a : f32
    %b32 = arith.addf %b31, %a : f32
    %b33 = arith.addf %b32, %a : f32
    %b34 = arith.addf %b33, %a : f32
    %b35 = arith.addf %b34, %a : f32
    %b36 = arith.addf %b35, %a : f32
    %b37 = arith.addf %b36, %a : f32
    %b38 = arith.addf %b37, %a : f32
    %b39 = arith.addf %b38, %a : f32
    %b40 = arith.addf %b39, %a : f32
    %b41 = arith.addf %b40, %a : f32
    %b42 = arith.addf %b41, %a : f32
    %b43 = arith.addf %b42, %a : f32
    %b44 = arith.addf %b43, %a : f32
    %b45 = arith.addf %b44, %a : f32
    %b46 = arith.addf %b45, %a : f32
    %b47 = arith.addf %b46, %a : f32
    %b48 = arith.addf %b47, %a : f32
    %b49 = arith.addf %b48, %a : f32
    %b50 = arith.addf %b49, %a : f32
    %b51 = arith.addf %b50, %a : f32
    %b52 = arith.addf %b51, %a : f32
    %b53 = arith.addf %b52, %a : f32
    %b54 = arith.addf %b53, %a : f32
    %b55 = arith.addf %b54, %a : f32
    %b56 = arith.addf %b55, %a : f32
    %b57 = arith.addf %b56, %a : f32
    %b58 = arith.addf %b57, %a : f32
    %b59 = arith.addf %b58, %a : f32
    %b60 = arith.addf %b59, %a : f32
    %b61 = arith.addf %b60, %a : f32
    %b62 = arith.addf %b61, %a : f32
    %b63 = arith.addf %b62, %a : f32
    %b64 = arith.addf %b63, %a : f32
    %b65 = arith.addf %b64, %a : f32
    %b66 = arith.addf %b65, %a : f32
    %b67 = arith.addf %b66, %a : f32
    %b68 = arith.addf %b67, %a : f32
    %b69 = arith.addf %b68, %a : f32
    %b70 = arith.addf %b69, %a : f32
    %b71 = arith.addf %b70, %a : f32
    %b72 = arith.addf %b71, %a : f32
    %b73 = arith.addf %b72, %a : f32
    %b74 = arith.addf %b73, %a : f32
    %b75 = arith.addf %b74, %a : f32
    %b76 = arith.addf %b75, %a : f32
    %b77 = arith.addf %b76, %a : f32
    %b78 = arith.addf %b77, %a : f32
    %b79 = arith.addf %b78, %a : f32
    %b80 = arith.addf %b79, %a : f32
    %b81 = arith.addf %b80, %a : f32
    %b82 = arith.addf %b81, %a : f32
    %b83 = arith.addf %b82, %a : f32
    %b84 = arith.addf %b83, %a : f32
    %b85 = arith.addf %b84, %a : f32
    %b86 = arith.addf %b85, %a : f32
    %b87 = arith.addf %b86, %a : f32
    %b88 = arith.addf %b87, %a : f32
    %b89 = arith.addf %b88, %a : f32
    %b90 = arith.addf %b89, %a : f32
    %b91 = arith.addf %b90, %a : f32
    %b92 = arith.addf %b91, %a : f32
    %b93 = arith.addf %b92, %a : f32
    %b94 = arith.addf %b93, %a : f32
    %b95 = arith.addf %b94, %a : f32
    %b96 = arith.addf %b95, %a : f32
    %b97 = arith.addf %b96, %a : f32
    %b98 = arith.addf %b97, %a : f32
    %b99 = arith.addf %b98, %a : f32
    %b100 = arith.addf %b99, %a : f32
    %b101 = arith.addf %b100, %a : f32
    %b102 = arith.addf %b101, %a : f32
    %b103 = arith.addf %b102, %a : f32
    %b104 = arith.addf %b103, %a : f32
    %b105 = arith.addf %b104, %a : f32
    %b106 = arith.addf %b105, %a : f32
    %b107 = arith.addf %b106, %a : f32
    %b108 = arith.addf %b107, %a : f32
    %b109 = arith.addf %b108, %a : f32
    %b110 = arith.addf %b109, %a : f32
    return %b110 : f32
  }
  func.func @entry_a(%a: f32) -> f32
      attributes { xla.backend_kind = #xla.backend_kind<cpu> } {
    %ret = xla.pure_call @big(%a) : (f32) -> (f32)
    return %ret : f32
  }
  func.func @entry_b(%a: f32) -> f32 {
    %ret = xla.pure_call @big(%a) : (f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @entry_a
// CHECK: xla.pure_call @big
// CHECK: @entry_b
// CHECK: xla.pure_call @big