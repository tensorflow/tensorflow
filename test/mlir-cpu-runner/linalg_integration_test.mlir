// RUN: mlir-opt %s -linalg-lower-to-llvm-dialect | mlir-cpu-runner -e dot -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-lower-to-loops -linalg-lower-to-llvm-dialect | mlir-cpu-runner -e dot -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-lower-to-llvm-dialect | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-lower-to-loops -linalg-lower-to-llvm-dialect | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s

func @linalg_dot_impl(%arg0 : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">,
                      %arg1 : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">,
                      %arg2 : !llvm<"{ float*, i64, [0 x i64], [0 x i64] }*">)

func @linalg_dot(%arg0 : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">,
                 %arg1 : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }">,
                 %arg2 : !llvm<"{ float*, i64, [0 x i64], [0 x i64] }">) {
  %c1 = llvm.constant(1) : !llvm.i64
  %0 = llvm.alloca %c1 x !llvm<"{ float*, i64, [1 x i64], [1 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">
  %1 = llvm.alloca %c1 x !llvm<"{ float*, i64, [1 x i64], [1 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">
  %2 = llvm.alloca %c1 x !llvm<"{ float*, i64, [0 x i64], [0 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, i64, [0 x i64], [0 x i64] }*">
  llvm.store %arg0, %0 : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">
  llvm.store %arg1, %1 : !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">
  llvm.store %arg2, %2 : !llvm<"{ float*, i64, [0 x i64], [0 x i64] }*">
  call @linalg_dot_impl(%0, %1, %2) : (!llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">, !llvm<"{ float*, i64, [1 x i64], [1 x i64] }*">, !llvm<"{ float*, i64, [0 x i64], [0 x i64] }*">) -> ()
  return
}

func @linalg_matmul_impl(%arg0 : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">,
                         %arg1 : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">,
                         %arg2 : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">)

func @linalg_matmul(%arg0 : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">,
                    %arg1 : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">,
                    %arg2 : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">) {
  %c1 = llvm.constant(1) : !llvm.i64
  %0 = llvm.alloca %c1 x !llvm<"{ float*, i64, [2 x i64], [2 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
  %1 = llvm.alloca %c1 x !llvm<"{ float*, i64, [2 x i64], [2 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
  %2 = llvm.alloca %c1 x !llvm<"{ float*, i64, [2 x i64], [2 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
  llvm.store %arg0, %0 : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
  llvm.store %arg1, %1 : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
  llvm.store %arg2, %2 : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">
  call @linalg_matmul_impl(%0, %1, %2) : (!llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">, !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">, !llvm<"{ float*, i64, [2 x i64], [2 x i64] }*">) -> ()
  return
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

func @dot() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c16 = constant 16 : index
  %f10 = constant 10.00000e+00 : f32
  %f1 = constant 1.00000e+00 : f32
  %f2 = constant 2.00000e+00 : f32

  %bA = call @alloc_filled_f32(%c16, %f2) : (index, f32) -> (!linalg.buffer<f32>)
  %bB = call @alloc_filled_f32(%c16, %f1) : (index, f32) -> (!linalg.buffer<f32>)
  %bC = call @alloc_filled_f32(%c1, %f10) : (index, f32) -> (!linalg.buffer<f32>)

  %R = linalg.range %c0:%c16:%c1 : !linalg.range
  %A = linalg.view %bA[%R] : !linalg.view<?xf32>
  %B = linalg.view %bB[%R] : !linalg.view<?xf32>
  %C = linalg.view %bC[] : !linalg.view<f32>

  linalg.dot(%A, %B, %C) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  %res = linalg.load %C[] : !linalg.view<f32>

  linalg.buffer_dealloc %bC : !linalg.buffer<f32>
  linalg.buffer_dealloc %bB : !linalg.buffer<f32>
  linalg.buffer_dealloc %bA : !linalg.buffer<f32>

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

  %bA = call @alloc_filled_f32(%c160, %f2) : (index, f32) -> (!linalg.buffer<f32>)
  %bB = call @alloc_filled_f32(%c160, %f1) : (index, f32) -> (!linalg.buffer<f32>)
  %bC = call @alloc_filled_f32(%c100, %f10) : (index, f32) -> (!linalg.buffer<f32>)

  %M = linalg.range %c0:%c10:%c1 : !linalg.range
  %N = linalg.range %c0:%c10:%c1 : !linalg.range
  %K = linalg.range %c0:%c16:%c1 : !linalg.range
  %A = linalg.view %bA[%M, %K] : !linalg.view<?x?xf32>
  %B = linalg.view %bB[%K, %N] : !linalg.view<?x?xf32>
  %C = linalg.view %bC[%M, %N] : !linalg.view<?x?xf32>

  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  %res = linalg.load %C[%c6, %c7] : !linalg.view<?x?xf32>

  linalg.buffer_dealloc %bC : !linalg.buffer<f32>
  linalg.buffer_dealloc %bB : !linalg.buffer<f32>
  linalg.buffer_dealloc %bA : !linalg.buffer<f32>

  return %res : f32
}


// All tests return this value
// CHECK: 4.2{{0+}}e+01
