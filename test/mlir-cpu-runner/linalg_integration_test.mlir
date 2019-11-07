// RUN: mlir-opt %s -convert-linalg-to-llvm | mlir-cpu-runner -e dot -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-lower-to-loops -convert-linalg-to-llvm | mlir-cpu-runner -e dot -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -convert-linalg-to-llvm | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-lower-to-loops -convert-linalg-to-llvm | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 -linalg-promote-subviews -linalg-lower-to-loops -convert-linalg-to-llvm | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 -linalg-promote-subviews -convert-linalg-to-llvm | mlir-cpu-runner -e matmul -entry-point-result=f32 -shared-libs=%linalg_test_lib_dir/libcblas%shlibext,%linalg_test_lib_dir/libcblas_interface%shlibext | FileCheck %s

#strided1D = (d0) -> (d0)
#strided2D = (d0, d1)[s0] -> (d0 * s0 + d1)

// Creates and returns a 1-D buffer of size %s filled with the value %f
func @alloc_filled_f32(%s : index, %f : f32) -> memref<?xi8> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %s4 = muli %s, %c4: index
  %buf = alloc(%s4) {alignment = 256} : memref<?xi8>
  %V = view %buf[%s][] : memref<?xi8> to memref<?xf32, #strided1D>
  linalg.fill(%V, %f) : memref<?xf32, #strided1D>, f32
  return %buf : memref<?xi8>
}

// Test for linalg.dot.
func @dot() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c16 = constant 16 : index
  %f10 = constant 10.00000e+00 : f32
  %f1 = constant 1.00000e+00 : f32
  %f2 = constant 2.00000e+00 : f32

  %bA = call @alloc_filled_f32(%c16, %f2) : (index, f32) -> (memref<?xi8>)
  %bB = call @alloc_filled_f32(%c16, %f1) : (index, f32) -> (memref<?xi8>)
  %bC = call @alloc_filled_f32(%c1, %f10) : (index, f32) -> (memref<?xi8>)

  %A = view %bA[%c16][] : memref<?xi8> to memref<?xf32, #strided1D>
  %B = view %bB[%c16][] : memref<?xi8> to memref<?xf32, #strided1D>
  %C = view %bC[][] : memref<?xi8> to memref<f32>

  linalg.dot(%A, %B, %C) : memref<?xf32, #strided1D>, memref<?xf32, #strided1D>, memref<f32>
  %res = load %C[] : memref<f32>

  dealloc %bC : memref<?xi8>
  dealloc %bB : memref<?xi8>
  dealloc %bA : memref<?xi8>

  return %res : f32
}

// Test for linalg.matmul.
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

  %bA = call @alloc_filled_f32(%c160, %f2) : (index, f32) -> (memref<?xi8>)
  %bB = call @alloc_filled_f32(%c160, %f1) : (index, f32) -> (memref<?xi8>)
  %bC = call @alloc_filled_f32(%c100, %f10) : (index, f32) -> (memref<?xi8>)

  %A = view %bA[%c10, %c16][] : memref<?xi8> to memref<?x?xf32, #strided2D>
  %B = view %bB[%c16, %c10][] : memref<?xi8> to memref<?x?xf32, #strided2D>
  %C = view %bC[%c10, %c10][] : memref<?xi8> to memref<?x?xf32, #strided2D>

  linalg.matmul(%A, %B, %C) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  %res = load %C[%c6, %c7] : memref<?x?xf32, #strided2D>

  dealloc %bC : memref<?xi8>
  dealloc %bB : memref<?xi8>
  dealloc %bA : memref<?xi8>

  return %res : f32
}

// All tests return this value
// CHECK: 4.2{{0+}}e+01
