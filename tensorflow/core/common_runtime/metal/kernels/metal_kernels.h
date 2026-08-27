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

// Elementwise arithmetic (Add, AddV2, Sub, Mul) and Cast, for float32 and
// float16, implemented as Metal compute shaders.
void RegisterMetalElementwiseKernels();

// MatMul for float32 and float16, backed by MPSMatrixMultiplication.
void RegisterMetalMatMulKernels();

// Conv2D and its gradients with respect to the input and the filter, on
// MPSGraph, for float32 and float16.
void RegisterMetalConvKernels();

// Relu, BiasAdd, Softmax, the dense and sparse softmax cross entropies, and
// the gradients of each, on MPSGraph.
void RegisterMetalNnKernels();

// MaxPool and its gradient, on MPSGraph.
void RegisterMetalPoolingKernels();

// Sum and Mean over float tensors, on MPSGraph. TensorFlow's DEVICE_DEFAULT
// registrations for these cover int32 in host memory only.
void RegisterMetalReductionKernels();

// Identity, which aliases its input when it can and blits when it cannot.
void RegisterMetalIdentityKernels();

// Registers every Metal kernel. Passed to core as the plugin's kernel module
// (PluggableDeviceInit_Api::init_kernel_fn), so core decides when kernel
// registration happens relative to device registration rather than the order
// falling out of static initialiser order.
void RegisterAllMetalKernels();

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNELS_H_
