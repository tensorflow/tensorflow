// RUN: mlir-opt -convert-gpu-to-spirv %s -o - | FileCheck %s

func @load_store(%arg0: memref<12x4xf32>, %arg1: memref<12x4xf32>, %arg2: memref<12x4xf32>) {
  %c0 = constant 0 : index
  %c12 = constant 12 : index
  %0 = subi %c12, %c0 : index
  %c1 = constant 1 : index
  %c0_0 = constant 0 : index
  %c4 = constant 4 : index
  %1 = subi %c4, %c0_0 : index
  %c1_1 = constant 1 : index
  %c1_2 = constant 1 : index
  "gpu.launch_func"(%0, %c1_2, %c1_2, %1, %c1_2, %c1_2, %arg0, %arg1, %arg2, %c0, %c0_0, %c1, %c1_1) {kernel = @load_store_kernel} : (index, index, index, index, index, index, memref<12x4xf32>, memref<12x4xf32>, memref<12x4xf32>, index, index, index, index) -> ()
  return
}

// CHECK-LABEL: spv.module "Logical" "GLSL450"
// CHECK: spv.globalVariable {{@.*}} bind(0, 0) : [[TYPE1:!spv.ptr<!spv.array<12 x !spv.array<4 x f32>>, StorageBuffer>]]
// CHECK-NEXT: spv.globalVariable {{@.*}} bind(0, 1) : [[TYPE2:!spv.ptr<!spv.array<12 x !spv.array<4 x f32>>, StorageBuffer>]]
// CHECK-NEXT: spv.globalVariable {{@.*}} bind(0, 2) : [[TYPE3:!spv.ptr<!spv.array<12 x !spv.array<4 x f32>>, StorageBuffer>]]
// CHECK: func @load_store_kernel([[ARG0:%.*]]: [[TYPE1]], [[ARG1:%.*]]: [[TYPE2]], [[ARG2:%.*]]: [[TYPE3]], [[ARG3:%.*]]: i32, [[ARG4:%.*]]: i32, [[ARG5:%.*]]: i32, [[ARG6:%.*]]: i32)
func @load_store_kernel(%arg0: memref<12x4xf32>, %arg1: memref<12x4xf32>, %arg2: memref<12x4xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: index)
  attributes  {gpu.kernel} {
  %0 = "gpu.block_id"() {dimension = "x"} : () -> index
  %1 = "gpu.block_id"() {dimension = "y"} : () -> index
  %2 = "gpu.block_id"() {dimension = "z"} : () -> index
  %3 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %4 = "gpu.thread_id"() {dimension = "y"} : () -> index
  %5 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %6 = "gpu.grid_dim"() {dimension = "x"} : () -> index
  %7 = "gpu.grid_dim"() {dimension = "y"} : () -> index
  %8 = "gpu.grid_dim"() {dimension = "z"} : () -> index
  %9 = "gpu.block_dim"() {dimension = "x"} : () -> index
  %10 = "gpu.block_dim"() {dimension = "y"} : () -> index
  %11 = "gpu.block_dim"() {dimension = "z"} : () -> index
  // CHECK: [[INDEX1:%.*]] = spv.IAdd [[ARG3]], {{%.*}}
  %12 = addi %arg3, %0 : index
  // CHECK: [[INDEX2:%.*]] = spv.IAdd [[ARG4]], {{%.*}}
  %13 = addi %arg4, %3 : index
  // CHECK: [[PTR1:%.*]] = spv.AccessChain [[ARG0]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}}
  // CHECK-NEXT: [[VAL1:%.*]] = spv.Load "StorageBuffer" [[PTR1]]
  %14 = load %arg0[%12, %13] : memref<12x4xf32>
  // CHECK: [[PTR2:%.*]] = spv.AccessChain [[ARG1]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}}
  // CHECK-NEXT: [[VAL2:%.*]] = spv.Load "StorageBuffer" [[PTR2]]
  %15 = load %arg1[%12, %13] : memref<12x4xf32>
  // CHECK: [[VAL3:%.*]] = spv.FAdd [[VAL1]], [[VAL2]]
  %16 = addf %14, %15 : f32
  // CHECK: [[PTR3:%.*]] = spv.AccessChain [[ARG2]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}}
  // CHECK-NEXT: spv.Store "StorageBuffer" [[PTR3]], [[VAL3]]
  store %16, %arg2[%12, %13] : memref<12x4xf32>
  return
}