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
// =============================================================================

// RUN: emitters_opt %s -split-input-file \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx90a\"}'" \
// RUN:   | FileCheck %s
// RUN: emitters_opt %s -split-input-file \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx942\"}'" \
// RUN:   | FileCheck %s
// RUN: emitters_opt %s -split-input-file \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx1100\"}'" \
// RUN:   | FileCheck %s
// RUN: emitters_opt %s -split-input-file \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx1200\"}'" \
// RUN:   | FileCheck %s
// RUN: emitters_opt %s -split-input-file \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx1250\"}'" \
// RUN:   | FileCheck %s

// bf16 exp and log are evaluated with the f32 transcendentals, so the lowering
// is the same on every AMD GPU; only the backend expansion of the final
// fptrunc differs. MathToROCDL has no bf16 lowering for them at all.

// exp(x) becomes exp2(x * log2(e)/2)^2, all in f32. Staying in f32 avoids
// rounding the scaled exponent to bf16, and halving the exponent keeps exp2
// out of the subnormal range it cannot produce; the squaring rounds back in.
module {
  func.func @exp_bf16(%arg0: bf16) -> bf16 {
    %0 = math.exp %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: llvm.func @exp_bf16
// CHECK: llvm.fpext {{.*}} : bf16 to f32
// CHECK: llvm.mlir.constant(0.72134751{{[0-9]*}} : f32)
// CHECK: llvm.fmul {{.*}} : f32
// CHECK: %[[ROOT:.*]] = llvm.call_intrinsic "llvm.amdgcn.exp2"({{.*}}) : (f32) -> f32
// CHECK: llvm.fmul %[[ROOT]], %[[ROOT]] : f32
// CHECK: llvm.fptrunc {{.*}} : f32 to bf16
// CHECK-NOT: __ocml

// -----

// log(x) becomes log2(x) * ln(2), all in f32. A subnormal argument would be
// flushed to zero by the hardware instruction and return -inf, so anything
// smaller in magnitude than the smallest normal f32 is scaled up by 2^64
// first and the result corrected by 64 * ln(2).
module {
  func.func @log_bf16(%arg0: bf16) -> bf16 {
    %0 = math.log %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: llvm.func @log_bf16
// CHECK: %[[X:.*]] = llvm.fpext {{.*}} : bf16 to f32
// CHECK: %[[ABS:.*]] = llvm.intr.fabs(%[[X]]) : (f32) -> f32
// CHECK: %[[SUB:.*]] = llvm.fcmp "olt" %[[ABS]], {{.*}} : f32
// CHECK: %[[SCALED:.*]] = llvm.fmul %[[X]], {{.*}} : f32
// CHECK: %[[ARG:.*]] = llvm.select %[[SUB]], %[[SCALED]], %[[X]] : i1, f32
// CHECK: llvm.call_intrinsic "llvm.amdgcn.log"(%[[ARG]]) : (f32) -> f32
// CHECK: llvm.fmul {{.*}} : f32
// CHECK: %[[CORR:.*]] = llvm.select %[[SUB]], {{.*}} : i1, f32
// CHECK: llvm.fsub {{.*}}, %[[CORR]] : f32
// CHECK: llvm.fptrunc {{.*}} : f32 to bf16
// CHECK-NOT: __ocml

// -----

// Vector bf16 exp and log are scalarized first by MathToROCDL (lower benefit),
// so each element takes the same f32 path.
module {
  func.func @exp_vector_bf16(%arg0: vector<2xbf16>) -> vector<2xbf16> {
    %0 = math.exp %arg0 : vector<2xbf16>
    return %0 : vector<2xbf16>
  }
}

// CHECK-LABEL: llvm.func @exp_vector_bf16
// CHECK-COUNT-2: llvm.call_intrinsic "llvm.amdgcn.exp2"({{.*}}) : (f32) -> f32
// CHECK-NOT: __ocml
