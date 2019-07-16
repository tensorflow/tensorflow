// RUN: mlir-opt %s -linalg-lower-to-llvm-dialect | mlir-cpu-runner -e dot -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-lower-to-loops -linalg-lower-to-llvm-dialect | mlir-cpu-runner -e dot -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-lower-to-llvm-dialect | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-lower-to-loops -linalg-lower-to-llvm-dialect | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s

func @fill_f32(%arg0 : !linalg.buffer<?xf32>, %f : f32) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %s = linalg.buffer_size %arg0 : !linalg.buffer<?xf32>
  %R = linalg.range %c0:%s:%c1 : !linalg.range
  %V = linalg.view %arg0[%R] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
  loop.for %i0 = %c0 to %s step %c1 {
    linalg.store %f, %V[%i0] : !linalg.view<?xf32>
  }
  return
}

func @alloc_filled_f32(%s : index, %f : f32) -> !linalg.buffer<?xf32> {
  %A = linalg.buffer_alloc %s : !linalg.buffer<?xf32>
  call @fill_f32(%A, %f) : (!linalg.buffer<?xf32>, f32) -> ()
  return %A : !linalg.buffer<?xf32>
}

func @dot() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c16 = constant 16 : index
  %f10 = constant 10.00000e+00 : f32
  %f1 = constant 1.00000e+00 : f32
  %f2 = constant 2.00000e+00 : f32

  %bA = call @alloc_filled_f32(%c16, %f2) : (index, f32) -> (!linalg.buffer<?xf32>)
  %bB = call @alloc_filled_f32(%c16, %f1) : (index, f32) -> (!linalg.buffer<?xf32>)
  %bC = call @alloc_filled_f32(%c1, %f10) : (index, f32) -> (!linalg.buffer<?xf32>)

  %R = linalg.range %c0:%c16:%c1 : !linalg.range
  %A = linalg.view %bA[%R] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
  %B = linalg.view %bB[%R] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
  %C = linalg.view %bC[] : !linalg.buffer<?xf32> -> !linalg.view<f32>

  linalg.dot(%A, %B, %C) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  %res = linalg.load %C[] : !linalg.view<f32>

  linalg.buffer_dealloc %bC : !linalg.buffer<?xf32>
  linalg.buffer_dealloc %bB : !linalg.buffer<?xf32>
  linalg.buffer_dealloc %bA : !linalg.buffer<?xf32>

  return %res : f32
}

func @matmul() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index
  %c10 = constant 10 : index
  %c16 = constant 16 : index
  %c100 = constant 100 : index
  %c160 = constant 160 : index
  %f1 = constant 1.00000e+00 : f32
  %f2 = constant 2.00000e+00 : f32
  %f10 = constant 10.00000e+00 : f32

  %bA = call @alloc_filled_f32(%c160, %f2) : (index, f32) -> (!linalg.buffer<?xf32>)
  %bB = call @alloc_filled_f32(%c160, %f1) : (index, f32) -> (!linalg.buffer<?xf32>)
  %bC = call @alloc_filled_f32(%c100, %f10) : (index, f32) -> (!linalg.buffer<?xf32>)

  %M = linalg.range %c0:%c10:%c1 : !linalg.range
  %N = linalg.range %c0:%c10:%c1 : !linalg.range
  %K = linalg.range %c0:%c16:%c1 : !linalg.range
  %A = linalg.view %bA[%M, %K] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  %B = linalg.view %bB[%K, %N] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  %C = linalg.view %bC[%M, %N] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>

  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  %res = linalg.load %C[%c6, %c7] : !linalg.view<?x?xf32>

  linalg.buffer_dealloc %bC : !linalg.buffer<?xf32>
  linalg.buffer_dealloc %bB : !linalg.buffer<?xf32>
  linalg.buffer_dealloc %bA : !linalg.buffer<?xf32>

  return %res : f32
}


// All tests return this value
// CHECK: 4.2{{0+}}e+01
