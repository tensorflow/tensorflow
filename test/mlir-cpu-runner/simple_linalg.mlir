// RUN: mlir-opt %s -linalg-lower-to-llvm-dialect | mlir-cpu-runner -e entry1 -entry-point-result=f32 | FileCheck %s

func @linalg_dot(!llvm<"{ float*, i64, [1 x i64], [1 x i64] }">,
                 !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">,
                 !llvm<"{ float*, i64, [0 x i64], [0 x i64] }">) {
  return
}

func @dot(%arg0: !linalg.buffer<f32>, %arg1: !linalg.buffer<f32>, %arg2: !linalg.buffer<f32>) -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %s = linalg.buffer_size %arg0 : !linalg.buffer<f32>
  %R = linalg.range %c0:%s:%c1 : !linalg.range
  %A = linalg.view %arg0[%R] : !linalg.view<?xf32>
  %B = linalg.view %arg1[%R] : !linalg.view<?xf32>
  %C = linalg.view %arg2[] : !linalg.view<f32>
  linalg.dot(%A, %B, %C) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  %res = linalg.load %C[] : !linalg.view<f32>
  return %res : f32
}

func @fill_f32(%arg0 : !linalg.buffer<f32>, %f : f32) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %s = linalg.buffer_size %arg0 : !linalg.buffer<f32>
  %R = linalg.range %c0:%s:%c1 : !linalg.range
  %V = linalg.view %arg0[%R] : !linalg.view<?xf32>
  affine.for %i0 = 0 to %s {
    linalg.store %f, %V[%i0] : !linalg.view<?xf32>
  }
  return
}

func @alloc_filled_f32(%s : index, %f : f32) -> !linalg.buffer<f32> {
  %A = linalg.buffer_alloc %s : !linalg.buffer<f32>
  call @fill_f32(%A, %f) : (!linalg.buffer<f32>, f32) -> ()
  return %A : !linalg.buffer<f32>
}

func @entry1() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c16 = constant 16 : index
  %f0 = constant 0.00000e+00 : f32
  %f1 = constant 0.00000e+00 : f32
  %f2 = constant 2.00000e+00 : f32

  %A = call @alloc_filled_f32(%c16, %f2) : (index, f32) -> (!linalg.buffer<f32>)
  %B = call @alloc_filled_f32(%c16, %f1) : (index, f32) -> (!linalg.buffer<f32>)
  %C = call @alloc_filled_f32(%c1, %f0) : (index, f32) -> (!linalg.buffer<f32>)
  %res = call @dot(%A, %B, %C) : (!linalg.buffer<f32>, !linalg.buffer<f32>, !linalg.buffer<f32>) -> (f32)
  linalg.buffer_dealloc %C : !linalg.buffer<f32>
  linalg.buffer_dealloc %B : !linalg.buffer<f32>
  linalg.buffer_dealloc %A : !linalg.buffer<f32>
  return %res : f32
}

// CHECK: 0.{{0+}}e+00