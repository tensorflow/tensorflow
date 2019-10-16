// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [5.356000e+03, 5.356000e+03, {{.*}}, 5.356000e+03, 5.356000e+03]
func @main() {
  %arg = alloc() : memref<13x4x2xf32>
  %dst = memref_cast %arg : memref<13x4x2xf32> to memref<?x?x?xf32>
  %zero = constant 0 : i32
  %one = constant 1 : index
  %sx = dim %dst, 0 : memref<?x?x?xf32>
  %sy = dim %dst, 1 : memref<?x?x?xf32>
  %sz = dim %dst, 2 : memref<?x?x?xf32>
  call @mcuMemHostRegister(%dst, %zero) : (memref<?x?x?xf32>, i32) -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %sy, %block_z = %sz)
             args(%kernel_dst = %dst) : memref<?x?x?xf32> {
    %t0 = muli %tz, %block_y : index
    %t1 = addi %ty, %t0 : index
    %t2 = muli %t1, %block_x : index
    %idx = addi %tx, %t2 : index
    %t3 = index_cast %idx : index to i32
    %val = sitofp %t3 : i32 to f32
    %sum = "gpu.all_reduce"(%val) ({}) { op = "add" } : (f32) -> (f32)
    store %sum, %kernel_dst[%tx, %ty, %tz] : memref<?x?x?xf32>
    gpu.return
  }
  call @mcuPrintFloat(%dst) : (memref<?x?x?xf32>) -> ()
  return
}

func @mcuMemHostRegister(%ptr : memref<?x?x?xf32>, %flags : i32)
func @mcuPrintFloat(%ptr : memref<?x?x?xf32>)
