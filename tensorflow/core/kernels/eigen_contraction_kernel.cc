/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/eigen_contraction_kernel.h"

#include <mutex>  // NOLINT(build/c++11)

#include "absl/base/call_once.h"

// We need a pair of compile time and runtime flags to disable compilation of
// custom contraction kernels for unsupported architectures (e.g. Android,
// iOS, ARM and PPC CPUs, etc...), and to be able to fallback on default Eigen
// matrix multiplication at runtime.
//
// It's not allowed to use absl flags library in Tensorflow, so we have to pass
// the configuration through the environment variable.
//
// Example:
//   bazel test \
//     --test_env=TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL=false \
//     //path/to:test

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)

namespace Eigen {
namespace internal {

// TODO(ezhulenev): This is a temporary workaround for disabling custom kernels
// at runtime in tests. We should always rely on compile time flags for that.
//
// Example:
//   bazel test \
//     --test_env=TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL=false \
//     //path/to:test
EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE bool UseCustomContractionKernels() {
  static bool use_custom_contraction_kernel = true;

// This subroutine should not be used in GPU. In case it is, a custom kernel
// should always be used
#if !defined __NVCC__ && !defined __HIP_DEVICE_COMPILE__
  static absl::once_flag initialized;
  absl::call_once(initialized, [&] {
    char* flag = std::getenv("TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL");
    if (flag && (strcmp(flag, "false") == 0 || strcmp(flag, "0") == 0)) {
      use_custom_contraction_kernel = false;
    }
  });
#endif

  return use_custom_contraction_kernel;
}

}  // namespace internal
}  // namespace Eigen
#endif
