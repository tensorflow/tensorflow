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

// Kernel-loader specs are structures that describe how to load a data-parallel
// kernel on a given platform for subsequent launching. Headers that instantiate
// these data structures will typically be auto-generated. However, users can
// also instantiate them by hand.
//
// A kernel with the same exact functionality and type signature may be
// implemented on several different platforms. Typical usage is to create a
// singleton that describes how to load a kernel on the various supported
// platforms:
//
//  static const MultiKernelLoaderSpec &SaxpySpec() {
//    static auto *mkls =
//        (new MultiKernelLoaderSpec{4 /* = arity */})
//            ->AddCudaPtxOnDisk(ptx_file_path, ptx_kernelname)
//            ->AddOpenCLTextOnDisk(opencl_text_file_path, ocl_kernelname);
//    };
//
//    return *mkls;
//  }
//
// This lazily instantiates an object that describes how to load CUDA PTX
// present on disk that implements saxpy for the for the CUDA platform, or
// OpenCL text present on disk that implements saxpy for an OpenCL-based
// platform. The CudaPtxOnDisk and OpenCLTextOnDisk objects are subtypes of
// KernelLoaderSpec -- KernelLoaderSpec describes how to load a kernel for
// subsequent launching on a single platform.
//
// For the loader functionality that accepts these KernelLoaderSpecs in order
// to grab the kernel appropriately, see StreamExecutor::GetKernel().

#ifndef TENSORFLOW_STREAM_EXECUTOR_KERNEL_SPEC_H_
#define TENSORFLOW_STREAM_EXECUTOR_KERNEL_SPEC_H_

#include "tensorflow/compiler/xla/stream_executor/kernel_spec.h"

#endif  // TENSORFLOW_STREAM_EXECUTOR_KERNEL_SPEC_H_
