// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-insert-pdl | FileCheck %s --check-prefix=INSERT
// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-insert-pdl \
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 9}'" \
// RUN: -xla-lower-pdl-wait \
// RUN: | FileCheck %s --check-prefix=LOWER

func.func @pdl_entry_store(%arg0: tensor<8xf32> {xla.slice_index = 0}) -> tensor<8xf32> attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %out = tensor.insert %cst into %arg0[%c0] : tensor<8xf32>
  func.return %out : tensor<8xf32>
}

// INSERT-LABEL: func.func @pdl_entry_store(
// INSERT: xla_gpu.pdl_wait
// INSERT: tensor.insert %cst into %arg0[%c0]
// INSERT: return

// LOWER-LABEL: func.func @pdl_entry_store(
// LOWER: nvvm.griddepcontrol wait
// LOWER-NOT: nvvm.griddepcontrol wait
// LOWER: llvm.store
// LOWER: return

// -----

func.func private @helper(%arg0: tensor<8xf32> {xla.slice_index = 0}, %arg1: index) -> f32 {
  %v = tensor.extract %arg0[%arg1] : tensor<8xf32>
  func.return %v : f32
}

func.func @entry_calls_helper(%arg0: tensor<8xf32> {xla.slice_index = 0}, %arg1: index) -> f32 attributes {xla.entry} {
  %v0 = call @helper(%arg0, %arg1) : (tensor<8xf32>, index) -> f32
  %v1 = tensor.extract %arg0[%arg1] : tensor<8xf32>
  %sum = arith.addf %v0, %v1 : f32
  func.return %sum : f32
}

// INSERT-LABEL: func.func private @helper(
// INSERT-NOT: pdl
// INSERT: tensor.extract
// INSERT: return
// INSERT-LABEL: func.func @entry_calls_helper(
// INSERT: xla_gpu.pdl_wait
// INSERT: call @helper
// INSERT: tensor.extract
// INSERT: return

// LOWER-LABEL: func.func private @helper(
// LOWER-NOT: griddepcontrol
// LOWER: llvm.load
// LOWER: return
// LOWER-LABEL: func.func @entry_calls_helper(
// LOWER: nvvm.griddepcontrol wait
// LOWER-NOT: nvvm.griddepcontrol wait
// LOWER: call @helper
// LOWER: llvm.load
// LOWER: return

// -----

func.func @mixed_memory_access(%arg0: tensor<4xf32> {xla.slice_index = 0}) -> (f32, f32) attributes {xla.entry} {
  %c0 = arith.constant 0 : index
  %t = arith.constant dense<1.0> : tensor<4xf32>
  %v_local = tensor.extract %t[%c0] : tensor<4xf32>
  %v_global = tensor.extract %arg0[%c0] : tensor<4xf32>
  func.return %v_local, %v_global : f32, f32
}

// INSERT-LABEL: func.func @mixed_memory_access
// INSERT: xla_gpu.pdl_wait
// INSERT: arith.constant 0
// INSERT: arith.constant dense
// INSERT: tensor.extract
// INSERT: tensor.extract
