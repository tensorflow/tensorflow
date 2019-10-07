// RUN: mlir-opt %s -linalg-lower-to-loops -linalg-convert-to-llvm | mlir-cpu-runner -e print_0d -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=PRINT-0D
// RUN: mlir-opt %s -linalg-lower-to-loops -linalg-convert-to-llvm | mlir-cpu-runner -e print_1d -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=PRINT-1D
// RUN: mlir-opt %s -linalg-lower-to-loops -linalg-convert-to-llvm | mlir-cpu-runner -e print_3d -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=PRINT-3D

func @print_0d() {
  %f = constant 2.00000e+00 : f32
  %A = alloc() : memref<f32>
  store %f, %A[]: memref<f32>
  call @print_memref_0d_f32(%A): (memref<f32>) -> ()
  dealloc %A : memref<f32>
  return
}

func @print_1d() {
  %f = constant 2.00000e+00 : f32
  %A = alloc() : memref<16xf32>
  %B = memref_cast %A: memref<16xf32> to memref<?xf32>
  linalg.fill(%B, %f) : memref<?xf32>, f32
  call @print_memref_1d_f32(%B): (memref<?xf32>) -> ()
  dealloc %A : memref<16xf32>
  return
}

func @print_3d() {
  %f = constant 2.00000e+00 : f32
  %f4 = constant 4.00000e+00 : f32
  %A = alloc() : memref<3x4x5xf32>
  %B = memref_cast %A: memref<3x4x5xf32> to memref<?x?x?xf32>
  linalg.fill(%B, %f) : memref<?x?x?xf32>, f32

  %c2 = constant 2 : index
  store %f4, %B[%c2, %c2, %c2]: memref<?x?x?xf32>

  call @print_memref_3d_f32(%B): (memref<?x?x?xf32>) -> ()
  dealloc %A : memref<3x4x5xf32>
  return
}

func @print_memref_0d_f32(memref<f32>)
func @print_memref_1d_f32(memref<?xf32>)
func @print_memref_3d_f32(memref<?x?x?xf32>)

// PRINT-0D: Memref base@ = {{.*}} rank = 0 offset = 0 data = [2]

// PRINT-1D: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [16] strides = [1] data =
// PRINT-1D-NEXT: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

// PRINT-3D: Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [3, 4, 5] strides = [20, 5, 1] data =
// PRINT-3D-COUNT-4: {{.*[[:space:]].*}}2,    2,    2,    2,    2
// PRINT-3D-COUNT-4: {{.*[[:space:]].*}}2,    2,    2,    2,    2
// PRINT-3D-COUNT-2: {{.*[[:space:]].*}}2,    2,    2,    2,    2
//    PRINT-3D-NEXT: 2,    2,    4,    2,    2
//    PRINT-3D-NEXT: 2,    2,    2,    2,    2
