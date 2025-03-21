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
  func.func private @fib0(%start : f32) -> f32 {
    %zero = arith.constant 0.0 : f32
    return %zero : f32
  }
  func.func private @fib1(%start : f32) -> f32 {
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

  func.func @caller(%a: f32) -> f32 {
    %ret = xla.pure_call @fib7(%a) : (f32) -> (f32)
    return %ret : f32
  }
}

// CHECK-LABEL: module {
// CHECK: @caller
// CHECK: arith.constant 0.000000e+00
// CHECK: xla.pure_call @fib5
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: xla.pure_call @fib5
// CHECK: arith.addf
// CHECK: arith.addf

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