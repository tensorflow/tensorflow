// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// gfx942: loads and stores; single-output stores; no loads when overwriting.
// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='rocm_compute_capability {gcn_arch_name: \"gfx942:sramecc+:xnack\"}' l2_cache_size=4194304" \
// RUN: | FileCheck %s --check-prefixes=CHECK-NT,CHECK-NT-LOADS,CHECK-NT-ONE-OUTPUT

// Other targets get no hints. gfx90a: excluded (not measured).
// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='rocm_compute_capability {gcn_arch_name: \"gfx90a:sramecc+:xnack\"}' l2_cache_size=4194304" \
// RUN: | FileCheck %s --check-prefix=CHECK-NT-OFF

// CUDA: excluded.
// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='cuda_compute_capability {major: 9}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-NT-OFF

// Streamed-once input of 16x L2 or more, contiguous output: load and store.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_copy(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.streamed_once,
                                   xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 1}, %i: index)
      -> tensor<16777216xf32> attributes {xla.contiguous_output, xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:          @nt_copy
// CHECK-NT-LOADS:            llvm.load {{.*}}nontemporal
// CHECK-NT:                  llvm.store {{.*}}nontemporal

// Other targets get no non-temporal hints.
// CHECK-NT-OFF-LABEL: @nt_copy
// CHECK-NT-OFF-NOT:     nontemporal

// -----

// Below 16x L2 (32 MiB): no hints, even if streamed once.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_streamed_below(
      %arg0: tensor<8388608xf32> {llvm.dereferenceable = 33554432 : index,
                                  xla.invariant, xla.streamed_once,
                                  xla.slice_index = 0},
      %arg1: tensor<8388608xf32> {llvm.dereferenceable = 33554432 : index,
                                  xla.slice_index = 1}, %i: index)
      -> tensor<8388608xf32> attributes {xla.contiguous_output, xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<8388608xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<8388608xf32>
    func.return %out : tensor<8388608xf32>
  }
}

// CHECK-NT-LABEL:  @nt_streamed_below
// CHECK-NT-NOT:      nontemporal
// CHECK-NT-OFF-LABEL: @nt_streamed_below
// CHECK-NT-OFF-NOT:    nontemporal

// -----

// Output below 16x L2: load hinted, store not.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_small_output(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.streamed_once,
                                   xla.slice_index = 0},
      %arg1: tensor<8388608xf32> {llvm.dereferenceable = 33554432 : index,
                                  xla.slice_index = 1}, %i: index)
      -> tensor<8388608xf32> attributes {xla.contiguous_output, xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<8388608xf32>
    func.return %out : tensor<8388608xf32>
  }
}

// CHECK-NT-LABEL:          @nt_small_output
// CHECK-NT-LOADS:            llvm.load {{.*}}nontemporal
// CHECK-NT-NOT:              llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_small_output
// CHECK-NT-OFF-NOT:     nontemporal

// -----

// Non-contiguous kernel: load hinted, store not.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_streamed_not_contiguous(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.streamed_once,
                                   xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 1}, %i: index)
      -> tensor<16777216xf32> attributes {xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:          @nt_streamed_not_contiguous
// CHECK-NT-LOADS:            llvm.load {{.*}}nontemporal
// CHECK-NT-NOT:              llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_streamed_not_contiguous
// CHECK-NT-OFF-NOT:     nontemporal

// -----

// Contiguous kernel reading the input more than once: no hints.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_contiguous_not_streamed(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.not_broadcast,
                                   xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 1}, %i: index, %j: index)
      -> tensor<16777216xf32> attributes {xla.contiguous_output, xla.entry} {
    %a = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %b = tensor.extract %arg0[%j] : tensor<16777216xf32>
    %s = arith.addf %a, %b : f32
    %out = tensor.insert %s into %arg1[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:  @nt_contiguous_not_streamed
// CHECK-NT-NOT:      nontemporal
// CHECK-NT-OFF-LABEL: @nt_contiguous_not_streamed
// CHECK-NT-OFF-NOT:    nontemporal

// -----

// Non-contiguous kernel, input may be broadcast: no hints.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_no_facts(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {xla.slice_index = 1}, %i: index)
      -> tensor<16777216xf32> attributes {xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:  @nt_no_facts
// CHECK-NT-NOT:      nontemporal
// CHECK-NT-OFF-LABEL: @nt_no_facts
// CHECK-NT-OFF-NOT:    nontemporal

// -----

// Non-contiguous kernel, input not broadcast: load hinted.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_not_broadcast(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.not_broadcast,
                                   xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {xla.slice_index = 1}, %i: index)
      -> tensor<16777216xf32> attributes {xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:          @nt_not_broadcast
// CHECK-NT-LOADS:            llvm.load {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_not_broadcast
// CHECK-NT-OFF-NOT:     nontemporal

// -----

// Below 16x L2: stays temporal.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_not_broadcast_below(
      %arg0: tensor<8388608xf32> {llvm.dereferenceable = 33554432 : index,
                                  xla.invariant, xla.not_broadcast,
                                  xla.slice_index = 0},
      %arg1: tensor<8388608xf32> {xla.slice_index = 1}, %i: index)
      -> tensor<8388608xf32> attributes {xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<8388608xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<8388608xf32>
    func.return %out : tensor<8388608xf32>
  }
}

// CHECK-NT-LABEL:  @nt_not_broadcast_below
// CHECK-NT-NOT:      nontemporal
// CHECK-NT-OFF-LABEL: @nt_not_broadcast_below
// CHECK-NT-OFF-NOT:    nontemporal

// -----

// Also read by another instruction: load hinted from 64x L2 (256 MiB).
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_read_elsewhere(
      %arg0: tensor<67108864xf32> {llvm.dereferenceable = 268435456 : index,
                                   xla.invariant, xla.read_elsewhere,
                                   xla.streamed_once, xla.slice_index = 0},
      %arg1: tensor<67108864xf32> {llvm.dereferenceable = 268435456 : index,
                                   xla.slice_index = 1}, %i: index)
      -> tensor<67108864xf32> attributes {xla.contiguous_output, xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<67108864xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<67108864xf32>
    func.return %out : tensor<67108864xf32>
  }
}

// CHECK-NT-LABEL:          @nt_read_elsewhere
// CHECK-NT-LOADS:            llvm.load {{.*}}nontemporal
// CHECK-NT:                  llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_read_elsewhere
// CHECK-NT-OFF-NOT:     nontemporal

// -----

// Below 64x L2: load stays temporal, store is hinted.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_read_elsewhere_below(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.read_elsewhere,
                                   xla.streamed_once, xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 1}, %i: index)
      -> tensor<16777216xf32> attributes {xla.contiguous_output, xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:  @nt_read_elsewhere_below
// CHECK-NT-NOT:      llvm.load {{.*}}nontemporal
// CHECK-NT:          llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_read_elsewhere_below
// CHECK-NT-OFF-NOT:    nontemporal

// -----

// Lines read in pieces: load stays temporal, store is hinted.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_piecewise_reads(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.piecewise_reads,
                                   xla.streamed_once, xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 1}, %i: index)
      -> tensor<16777216xf32> attributes {xla.contiguous_output, xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %out = tensor.insert %v into %arg1[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:  @nt_piecewise_reads
// CHECK-NT-NOT:      llvm.load {{.*}}nontemporal
// CHECK-NT:          llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_piecewise_reads
// CHECK-NT-OFF-NOT:    nontemporal

// -----

// Output read back by the kernel: store stays temporal.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_output_read_back(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.streamed_once,
                                   xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 1}, %i: index)
      -> tensor<16777216xf32> attributes {xla.contiguous_output, xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %prev = tensor.extract %arg1[%i] : tensor<16777216xf32>
    %s = arith.addf %v, %prev : f32
    %out = tensor.insert %s into %arg1[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:          @nt_output_read_back
// CHECK-NT-LOADS:            llvm.load {{.*}}nontemporal
// CHECK-NT-NOT:              llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_output_read_back
// CHECK-NT-OFF-NOT:     nontemporal

// -----

// In-place output (AdamW): no load hint on gfx942; store still hinted.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_in_place_output(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.streamed_once,
                                   xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.aliased, xla.slice_index = 1},
      %arg2: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.aliased, xla.slice_index = 1}, %i: index)
      -> tensor<16777216xf32> attributes {xla.contiguous_output, xla.entry} {
    %g = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %w = tensor.extract %arg1[%i] : tensor<16777216xf32>
    %s = arith.addf %g, %w : f32
    %out = tensor.insert %s into %arg2[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:  @nt_in_place_output
// CHECK-NT-NOT:      llvm.load {{.*}}nontemporal
// CHECK-NT:          llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_in_place_output
// CHECK-NT-OFF-NOT:    nontemporal

// -----

// Two large outputs: gfx942 hints the load, not the stores.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_two_outputs(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.streamed_once,
                                   xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 1},
      %arg2: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 2}, %i: index)
      -> (tensor<16777216xf32>, tensor<16777216xf32>)
      attributes {xla.contiguous_output, xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %out0 = tensor.insert %v into %arg1[%i] : tensor<16777216xf32>
    %out1 = tensor.insert %v into %arg2[%i] : tensor<16777216xf32>
    func.return %out0, %out1 : tensor<16777216xf32>, tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:              @nt_two_outputs
// CHECK-NT-LOADS:                llvm.load {{.*}}nontemporal
// CHECK-NT-ONE-OUTPUT-NOT:       llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_two_outputs
// CHECK-NT-OFF-NOT:     nontemporal

// -----

// Only outputs of 16x L2 or more count: the large store is hinted.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @nt_one_large_output(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.streamed_once,
                                   xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 1},
      %arg2: tensor<8388608xf32> {llvm.dereferenceable = 33554432 : index,
                                  xla.slice_index = 2}, %i: index)
      -> (tensor<16777216xf32>, tensor<8388608xf32>)
      attributes {xla.contiguous_output, xla.entry} {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    %out0 = tensor.insert %v into %arg1[%i] : tensor<16777216xf32>
    %out1 = tensor.insert %v into %arg2[%i] : tensor<8388608xf32>
    func.return %out0, %out1 : tensor<16777216xf32>, tensor<8388608xf32>
  }
}

// CHECK-NT-LABEL:          @nt_one_large_output
// CHECK-NT-LOADS:            llvm.load {{.*}}nontemporal
// CHECK-NT:                  llvm.store {{.*}}nontemporal
// CHECK-NT-NOT:              llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_one_large_output
// CHECK-NT-OFF-NOT:     nontemporal

// -----

// A load in a called function: its arg is sized by the entry's slice. The
// facts are removed.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func private @nt_helper(
      %arg0: tensor<16777216xf32> {xla.invariant, xla.streamed_once,
                                   xla.slice_index = 0}, %i: index) -> f32 {
    %v = tensor.extract %arg0[%i] : tensor<16777216xf32>
    func.return %v : f32
  }

  func.func @nt_entry_with_helper(
      %arg0: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.invariant, xla.streamed_once,
                                   xla.slice_index = 0},
      %arg1: tensor<16777216xf32> {llvm.dereferenceable = 67108864 : index,
                                   xla.slice_index = 1}, %i: index)
      -> tensor<16777216xf32> attributes {xla.contiguous_output, xla.entry} {
    %v = func.call @nt_helper(%arg0, %i) : (tensor<16777216xf32>, index) -> f32
    %out = tensor.insert %v into %arg1[%i] : tensor<16777216xf32>
    func.return %out : tensor<16777216xf32>
  }
}

// CHECK-NT-LABEL:          @nt_helper
// CHECK-NT-NOT:              xla.streamed_once
// CHECK-NT-LOADS:            llvm.load {{.*}}nontemporal
// CHECK-NT-LABEL:          @nt_entry_with_helper
// CHECK-NT-NOT:              xla.streamed_once
// CHECK-NT-NOT:              xla.contiguous_output
// CHECK-NT:                  llvm.store {{.*}}nontemporal
// CHECK-NT-OFF-LABEL: @nt_helper
// CHECK-NT-OFF-NOT:     xla.streamed_once
// CHECK-NT-OFF-NOT:     nontemporal
// CHECK-NT-OFF-LABEL: @nt_entry_with_helper
// CHECK-NT-OFF-NOT:     xla.streamed_once
// CHECK-NT-OFF-NOT:     xla.contiguous_output
// CHECK-NT-OFF-NOT:     nontemporal
