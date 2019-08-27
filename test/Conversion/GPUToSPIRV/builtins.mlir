// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv %s -o - | FileCheck %s

func @builtin() {
  %c0 = constant 1 : index
  "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = @builtin_workgroup_id_x} : (index, index, index, index, index, index) -> ()
  return
}

// CHECK-LABEL:  spv.module "Logical" "VulkanKHR"
// CHECK: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
func @builtin_workgroup_id_x()
  attributes {gpu.kernel} {
  // CHECK: [[ADDRESS:%.*]] = spv._address_of [[WORKGROUPID]]
  // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
  // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
  %0 = "gpu.block_id"() {dimension = "x"} : () -> index
  return
}

// -----

func @builtin() {
  %c0 = constant 1 : index
  "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = @builtin_workgroup_id_y} : (index, index, index, index, index, index) -> ()
  return
}

// CHECK-LABEL:  spv.module "Logical" "VulkanKHR"
// CHECK: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
func @builtin_workgroup_id_y()
  attributes {gpu.kernel} {
  // CHECK: [[ADDRESS:%.*]] = spv._address_of [[WORKGROUPID]]
  // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
  // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}1 : i32{{\]}}
  %0 = "gpu.block_id"() {dimension = "y"} : () -> index
  return
}

// -----

func @builtin() {
  %c0 = constant 1 : index
  "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = @builtin_workgroup_id_z} : (index, index, index, index, index, index) -> ()
  return
}

// CHECK-LABEL:  spv.module "Logical" "VulkanKHR"
// CHECK: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
func @builtin_workgroup_id_z()
  attributes {gpu.kernel} {
  // CHECK: [[ADDRESS:%.*]] = spv._address_of [[WORKGROUPID]]
  // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
  // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}2 : i32{{\]}}
  %0 = "gpu.block_id"() {dimension = "z"} : () -> index
  return
}

// -----

func @builtin() {
  %c0 = constant 1 : index
  "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = @builtin_workgroup_size_x} : (index, index, index, index, index, index) -> ()
  return
}

// CHECK-LABEL:  spv.module "Logical" "VulkanKHR"
// CHECK: spv.globalVariable [[WORKGROUPSIZE:@.*]] built_in("WorkgroupSize")
func @builtin_workgroup_size_x()
  attributes {gpu.kernel} {
  // CHECK: [[ADDRESS:%.*]] = spv._address_of [[WORKGROUPSIZE]]
  // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
  // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
  %0 = "gpu.block_dim"() {dimension = "x"} : () -> index
  return
}

// -----

func @builtin() {
  %c0 = constant 1 : index
  "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = @builtin_local_id_x} : (index, index, index, index, index, index) -> ()
  return
}

// CHECK-LABEL:  spv.module "Logical" "VulkanKHR"
// CHECK: spv.globalVariable [[LOCALINVOCATIONID:@.*]] built_in("LocalInvocationId")
func @builtin_local_id_x()
  attributes {gpu.kernel} {
  // CHECK: [[ADDRESS:%.*]] = spv._address_of [[LOCALINVOCATIONID]]
  // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
  // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
  %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
  return
}

// -----

func @builtin() {
  %c0 = constant 1 : index
  "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = @builtin_num_workgroups_x} : (index, index, index, index, index, index) -> ()
  return
}

// CHECK-LABEL:  spv.module "Logical" "VulkanKHR"
// CHECK: spv.globalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups")
func @builtin_num_workgroups_x()
  attributes {gpu.kernel} {
  // CHECK: [[ADDRESS:%.*]] = spv._address_of [[NUMWORKGROUPS]]
  // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
  // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
  %0 = "gpu.grid_dim"() {dimension = "x"} : () -> index
  return
}
