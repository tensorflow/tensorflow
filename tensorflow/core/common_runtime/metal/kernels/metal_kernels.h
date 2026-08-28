/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNELS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNELS_H_

// Names no Objective-C types; includable from plain C++.

namespace tensorflow {
namespace metal {

// Identity, Snapshot and the other pass-through forms.
void RegisterMetalIdentityKernels();

// The unary and binary arithmetic, with NumPy broadcasting.
void RegisterMetalElementwiseKernels();

// MatMul, through MPSMatrixMultiplication.
void RegisterMetalMatMulKernels();

// Every kernel this backend provides. Called by the plugin registrar through
// the PluggableDevice kernel module, so that core orders kernel registration
// against device registration rather than leaving it to link order.
void RegisterAllMetalKernels();

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNELS_H_
