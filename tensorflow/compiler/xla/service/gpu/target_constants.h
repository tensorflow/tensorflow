/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TARGET_CONSTANTS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TARGET_CONSTANTS_H_

namespace xla {
namespace gpu {

namespace nvptx {
// The triple that represents our target.
constexpr char kTargetTriple[] = "nvptx64-nvidia-cuda";

// The data layout of the emitted module. Copied from computeDataLayout in
// NVPTXTargetMachine.cpp.
constexpr char kDataLayout[] = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";
}  // namespace nvptx

namespace amdgpu {

// The triple that represents our target on LLVM AMDGPU backend.
constexpr char kTargetTriple[] = "amdgcn-amd-amdhsa";

// The data layout of the emitted module.
constexpr char kDataLayout[] =
    "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32"
    "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
    "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5";

}  // namespace amdgpu

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TARGET_CONSTANTS_H_
