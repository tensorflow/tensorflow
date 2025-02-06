/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_TARGET_CONSTANTS_H_
#define XLA_SERVICE_GPU_TARGET_CONSTANTS_H_

namespace xla {
namespace gpu {

namespace nvptx {
// The triple that represents our target.
inline const char* TargetTriple() {
  static constexpr char kTargetTriple[] = "nvptx64-nvidia-cuda";
  return kTargetTriple;
}

// The data layout of the emitted module. Copied from computeDataLayout in
// NVPTXTargetMachine.cpp.
inline const char* DataLayout() {
  static constexpr char kDataLayout[] =
      "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";
  return kDataLayout;
}
}  // namespace nvptx

namespace amdgpu {

// The triple that represents our target on LLVM AMDGPU backend.
inline const char* TargetTriple() {
  static constexpr char kTargetTriple[] = "amdgcn-amd-amdhsa";
  return kTargetTriple;
}

// The data layout of the emitted module.
inline const char* DataLayout() {
  static constexpr char kDataLayout[] =
      "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32"
      "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
      "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5";
  return kDataLayout;
}

}  // namespace amdgpu

namespace spir {
// The triple that represents our target on SPIR backend.
inline const char* TargetTriple() {
  static constexpr char kTargetTriple[] = "spir64-unknown-unknown";
  return kTargetTriple;
}

// The data layout of the emitted module.
inline const char* DataLayout() {
  static constexpr char kDataLayout[] =
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:"
      "32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:"
      "128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:"
      "1024";
  return kDataLayout;
}
}  // namespace spir

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TARGET_CONSTANTS_H_
