/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_PORT_H_
#define TENSORFLOW_CORE_UTIL_PORT_H_

namespace tensorflow {

// Returns true if GOOGLE_CUDA is defined.
bool IsGoogleCudaEnabled();

// Returns true if TENSORFLOW_USE_ROCM is defined. (i.e. TF is built with ROCm)
bool IsBuiltWithROCm();

// Returns true if TENSORFLOW_USE_XLA is defined. (i.e. TF is built with XLA)
bool IsBuiltWithXLA();

// Returns true if TENSORFLOW_USE_NVCC is defined. (i.e. TF is built with nvcc)
bool IsBuiltWithNvcc();

// Returns true if TF_LLVM_AARCH32_AVAILABLE is defined. (i.e. TF is built for
// AArch32)
bool IsAArch32Available();

// Returns true if TF_LLVM_AARCH64_AVAILABLE is defined. (i.e. TF is built for
// AArch64)
bool IsAArch64Available();

// Returns true if TF_LLVM_POWERPC_AVAILABLE is defined. (i.e. TF is built for
// PowerPC)
bool IsPowerPCAvailable();

// Returns true if TF_LLVM_S390X_AVAILABLE is defined. (i.e. TF is built for
// System z)
bool IsSystemZAvailable();

// Returns true if TF_LLVM_X86_AVAILABLE is defined. (i.e. TF is built for X86)
bool IsX86Available();

// Returns true if either
//
//   GOOGLE_CUDA is defined, and the given CUDA version supports
//   half-precision matrix multiplications and convolution operations.
//
//     OR
//
//   TENSORFLOW_USE_ROCM is defined
//
bool GpuSupportsHalfMatMulAndConv();

// Returns true if INTEL_MKL is defined
bool IsMklEnabled();

// Returns true if TF_ENABLE_ZENDNN_OPTS is set to 1
bool IsZenDnnEnabled();

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_PORT_H_
