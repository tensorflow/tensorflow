// RUN: mlir_fusions_opt %s -split-input-file -inline | FileCheck %s

module {
  func.func private @mul(%a: f32, %b: f32) -> f32 {
    %ret = arith.mulf %a, %b : f32
    return %ret : f32
  }

  func.func private @add(%a: f32, %b: f32) -> f32 {
    %add = arith.addf %a, %b : f32
    %ret = xla_gpu.pure_call @mul(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %ret = xla_gpu.pure_call @add(%a, %b) : (f32, f32) -> (f32)
    return %ret : f32
  }
}
// CHECK: @caller
// CHECK-NOT: xla_gpu.pure_call @add
// CHECK: arith.addf
// CHECK-NOT: xla_gpu.pure_call @mul
// CHECK: arith.mulf

// -----

module {
  func.func @fused_computation(%arg0: tensor<2xf32> {xla.slice_index = 0 : index}, %arg1: tensor<2xf32> {xla.slice_index = 1 : index}, %arg2: tensor<2xf32> {xla.slice_index = 2 : index}) -> tensor<2xf32> attributes {xla.entry} {
    %0 = gpu.thread_id  x {xla.range = [0 : index, 1 : index]}
    %1 = xla_gpu.pure_call @fused_computation_atan2(%arg0, %arg1, %0) : (tensor<2xf32>, tensor<2xf32>, index) -> f32
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
// CHECK: @fused_computation
// CHECK-NOT: xla_gpu.pure_call @add
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
    %ret = xla_gpu.pure_call @large(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %add = xla_gpu.pure_call @add(%a, %b) : (f32, f32) -> (f32)
    %ret = xla_gpu.pure_call @large(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }
}
// CHECK: @caller
// CHECK: arith.addf
// CHECK: xla_gpu.pure_call @large
// CHECK: xla_gpu.pure_call @large

// -----

module {
  func.func private @add(%a: f32, %b: f32) -> f32 {
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %add = xla_gpu.pure_call @add(%a, %b) : (f32, f32) -> (f32)
    %ret = xla_gpu.pure_call @add(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }
}
// CHECK: @caller
// CHECK-NOT: xla_gpu.pure_call
// CHECK: arith.addf
// CHECK: arith.addf

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
    %a = xla_gpu.pure_call @fib0(%start) : (f32) -> (f32)
    %b = xla_gpu.pure_call @fib1(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @fib3(%start : f32) -> f32 {
    %a = xla_gpu.pure_call @fib1(%start) : (f32) -> (f32)
    %b = xla_gpu.pure_call @fib2(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @fib4(%start : f32) -> f32 {
    %a = xla_gpu.pure_call @fib2(%start) : (f32) -> (f32)
    %b = xla_gpu.pure_call @fib3(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  // When inlining the other functions into @fib5, this function exceeds the
  // threshold for inlining.
  func.func private @fib5(%start : f32) -> f32 {
    %a = xla_gpu.pure_call @fib3(%start) : (f32) -> (f32)
    %b = xla_gpu.pure_call @fib4(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  // As we do not inline @fib5 into @fib6, this function stays below the
  // threshold for inlining.
  func.func private @fib6(%start : f32) -> f32 {
    %a = xla_gpu.pure_call @fib4(%start) : (f32) -> (f32)
    %b = xla_gpu.pure_call @fib5(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }
  func.func private @fib7(%start : f32) -> f32 {
    %a = xla_gpu.pure_call @fib5(%start) : (f32) -> (f32)
    %b = xla_gpu.pure_call @fib6(%start) : (f32) -> (f32)
    %ret = arith.addf %a, %b : f32
    return %ret : f32
  }

  func.func @caller(%a: f32) -> f32 {
    %ret = xla_gpu.pure_call @fib7(%a) : (f32) -> (f32)
    return %ret : f32
  }
}
// CHECK: @caller
// CHECK: arith.constant 0.000000e+00
// CHECK: xla_gpu.pure_call @fib5
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: xla_gpu.pure_call @fib5
// CHECK: arith.addf
// CHECK: arith.addf

// -----

module {
  func.func private @complex(%a: f32, %b: f32) -> complex<f32> {
    %ret = complex.create %a, %b : complex<f32>
    return %ret : complex<f32>
  }

  func.func @caller(%a: f32, %b: f32) -> complex<f32> {
    %ret = xla_gpu.pure_call @complex(%a, %b) : (f32, f32) -> (complex<f32>)
    return %ret : complex<f32>
  }
}

// CHECK: @caller
// CHECK-NEXT: complex.create
