// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @memory_barrier_0() -> () {
    // CHECK: spv.MemoryBarrier "Device", "Release|UniformMemory"
    spv.MemoryBarrier "Device", "Release|UniformMemory"
    spv.Return
  }
  func @memory_barrier_1() -> () {
    // CHECK: spv.MemoryBarrier "Subgroup", "AcquireRelease|SubgroupMemory"
    spv.MemoryBarrier "Subgroup", "AcquireRelease|SubgroupMemory"
    spv.Return
  }
  func @control_barrier_0() -> () {
    // CHECK: spv.ControlBarrier "Device", "Workgroup", "Release|UniformMemory"
    spv.ControlBarrier "Device", "Workgroup", "Release|UniformMemory"
    spv.Return
  }
  func @control_barrier_1() -> () {
    // CHECK: spv.ControlBarrier "Workgroup", "Invocation", "AcquireRelease|UniformMemory"
    spv.ControlBarrier "Workgroup", "Invocation", "AcquireRelease|UniformMemory"
    spv.Return
  }
}
