// RUN: mlir-opt -convert-gpu-to-spirv %s -o - | FileCheck %s

module attributes {gpu.container_module} {
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
    "gpu.launch_func"(%0, %c1_2, %c1_2, %1, %c1_2, %c1_2, %arg0, %arg1, %arg2, %c0, %c0_0, %c1, %c1_1) {kernel = "load_store_kernel", kernel_module = @kernels} : (index, index, index, index, index, index, memref<12x4xf32>, memref<12x4xf32>, memref<12x4xf32>, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL: spv.module "Logical" "GLSL450"
  module @kernels attributes {gpu.kernel_module} {
    // CHECK-DAG: spv.globalVariable [[WORKGROUPSIZEVAR:@.*]] built_in("WorkgroupSize") : !spv.ptr<vector<3xi32>, Input>
    // CHECK-DAG: spv.globalVariable [[NUMWORKGROUPSVAR:@.*]] built_in("NumWorkgroups") : !spv.ptr<vector<3xi32>, Input>
    // CHECK-DAG: spv.globalVariable [[LOCALINVOCATIONIDVAR:@.*]] built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    // CHECK-DAG: spv.globalVariable [[WORKGROUPIDVAR:@.*]] built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
    // CHECK-DAG: spv.globalVariable [[VAR0:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32 [4]> [16]> [0]>, StorageBuffer>
    // CHECK-DAG: spv.globalVariable [[VAR1:@.*]] bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32 [4]> [16]> [0]>, StorageBuffer>
    // CHECK-DAG: spv.globalVariable [[VAR2:@.*]] bind(0, 2) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32 [4]> [16]> [0]>, StorageBuffer>
    // CHECK-DAG: spv.globalVariable [[VAR3:@.*]] bind(0, 3) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    // CHECK-DAG: spv.globalVariable [[VAR4:@.*]] bind(0, 4) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    // CHECK-DAG: spv.globalVariable [[VAR5:@.*]] bind(0, 5) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    // CHECK-DAG: spv.globalVariable [[VAR6:@.*]] bind(0, 6) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    // CHECK: func [[FN:@.*]]()
    func @load_store_kernel(%arg0: memref<12x4xf32>, %arg1: memref<12x4xf32>, %arg2: memref<12x4xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: index)
      attributes  {gpu.kernel} {
      // CHECK: [[ADDRESSARG0:%.*]] = spv._address_of [[VAR0]]
      // CHECK: [[CONST0:%.*]] = spv.constant 0 : i32
      // CHECK: [[ARG0:%.*]] = spv.AccessChain [[ADDRESSARG0]]{{\[}}[[CONST0]]
      // CHECK: [[ADDRESSARG1:%.*]] = spv._address_of [[VAR1]]
      // CHECK: [[CONST1:%.*]] = spv.constant 0 : i32
      // CHECK: [[ARG1:%.*]] = spv.AccessChain [[ADDRESSARG1]]{{\[}}[[CONST1]]
      // CHECK: [[ADDRESSARG2:%.*]] = spv._address_of [[VAR2]]
      // CHECK: [[CONST2:%.*]] = spv.constant 0 : i32
      // CHECK: [[ARG2:%.*]] = spv.AccessChain [[ADDRESSARG2]]{{\[}}[[CONST2]]
      // CHECK: [[ADDRESSARG3:%.*]] = spv._address_of [[VAR3]]
      // CHECK: [[CONST3:%.*]] = spv.constant 0 : i32
      // CHECK: [[ARG3PTR:%.*]] = spv.AccessChain [[ADDRESSARG3]]{{\[}}[[CONST3]]
      // CHECK: [[ARG3:%.*]] = spv.Load "StorageBuffer" [[ARG3PTR]]
      // CHECK: [[ADDRESSARG4:%.*]] = spv._address_of [[VAR4]]
      // CHECK: [[CONST4:%.*]] = spv.constant 0 : i32
      // CHECK: [[ARG4PTR:%.*]] = spv.AccessChain [[ADDRESSARG4]]{{\[}}[[CONST4]]
      // CHECK: [[ARG4:%.*]] = spv.Load "StorageBuffer" [[ARG4PTR]]
      // CHECK: [[ADDRESSARG5:%.*]] = spv._address_of [[VAR5]]
      // CHECK: [[CONST5:%.*]] = spv.constant 0 : i32
      // CHECK: [[ARG5PTR:%.*]] = spv.AccessChain [[ADDRESSARG5]]{{\[}}[[CONST5]]
      // CHECK: [[ARG5:%.*]] = spv.Load "StorageBuffer" [[ARG5PTR]]
      // CHECK: [[ADDRESSARG6:%.*]] = spv._address_of [[VAR6]]
      // CHECK: [[CONST6:%.*]] = spv.constant 0 : i32
      // CHECK: [[ARG6PTR:%.*]] = spv.AccessChain [[ADDRESSARG6]]{{\[}}[[CONST6]]
      // CHECK: [[ARG6:%.*]] = spv.Load "StorageBuffer" [[ARG6PTR]]
      // CHECK: [[ADDRESSWORKGROUPID:%.*]] = spv._address_of [[WORKGROUPIDVAR]]
      // CHECK: [[WORKGROUPID:%.*]] = spv.Load "Input" [[ADDRESSWORKGROUPID]]
      // CHECK: [[WORKGROUPIDX:%.*]] = spv.CompositeExtract [[WORKGROUPID]]{{\[}}0 : i32{{\]}}
      // CHECK: [[ADDRESSLOCALINVOCATIONID:%.*]] = spv._address_of [[LOCALINVOCATIONIDVAR]]
      // CHECK: [[LOCALINVOCATIONID:%.*]] = spv.Load "Input" [[ADDRESSLOCALINVOCATIONID]]
      // CHECK: [[LOCALINVOCATIONIDX:%.*]] = spv.CompositeExtract [[LOCALINVOCATIONID]]{{\[}}0 : i32{{\]}}
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
      // CHECK: [[INDEX1:%.*]] = spv.IAdd [[ARG3]], [[WORKGROUPIDX]]
      %12 = addi %arg3, %0 : index
      // CHECK: [[INDEX2:%.*]] = spv.IAdd [[ARG4]], [[LOCALINVOCATIONIDX]]
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
  }
}
