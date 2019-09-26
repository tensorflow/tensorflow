// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [8.128000e+03, 8.128000e+03, {{.*}}, 8.128000e+03, 8.128000e+03]
func @main() {
  %arg = alloc() : memref<128xf32>
  %dst = memref_cast %arg : memref<128xf32> to memref<?xf32>
  %zero = constant 0 : i32
  %one = constant 1 : index
  %size = dim %dst, 0 : memref<?xf32>
  call @mcuMemHostRegister(%dst, %zero) : (memref<?xf32>, i32) -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %size, %block_y = %one, %block_z = %one)
             args(%kernel_dst = %dst) : memref<?xf32> {
    %idx = index_cast %tx : index to i32
    %val = sitofp %idx : i32 to f32
    %sum = "gpu.all_reduce"(%val) { op = "add" } : (f32) -> (f32)
    store %sum, %kernel_dst[%tx] : memref<?xf32>
    gpu.return
  }
  call @mcuPrintFloat(%dst) : (memref<?xf32>) -> ()
  return
}

func @mcuMemHostRegister(%ptr : memref<?xf32>, %flags : i32)
func @mcuPrintFloat(%ptr : memref<?xf32>)
