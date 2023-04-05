/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUDA_TEST_H_
#define TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUDA_TEST_H_

#include "tensorflow/compiler/xla/backends/profiler/gpu/cuda_test.h"

namespace tensorflow {
namespace profiler {
namespace test {

using xla::profiler::test::EmptyKernel;          // NOLINT
using xla::profiler::test::MemCopyD2H;           // NOLINT
using xla::profiler::test::MemCopyH2D;           // NOLINT
using xla::profiler::test::MemCopyH2D_Async;     // NOLINT
using xla::profiler::test::MemCopyP2PAvailable;  // NOLINT
using xla::profiler::test::MemCopyP2PExplicit;   // NOLINT
using xla::profiler::test::MemCopyP2PImplicit;   // NOLINT
using xla::profiler::test::PrintfKernel;         // NOLINT
using xla::profiler::test::Synchronize;          // NOLINT

}  // namespace test
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUDA_TEST_H_
