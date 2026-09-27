/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCSOLVER_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCSOLVER_WRAPPER_H_

#include "rocm/include/rocsolver/rocsolver.h"
#include "rocm/rocm_config.h"

namespace stream_executor {
namespace wrap {

#define ROCSOLVER_API_WRAPPER(api_name) using ::api_name;

// clang-format off
#define FOREACH_ROCSOLVER_API(__macro)      \
  __macro(rocsolver_cgetrf)                 \
  __macro(rocsolver_dgetrf)                 \
  __macro(rocsolver_sgetrf)                 \
  __macro(rocsolver_zgetrf)                 \
  __macro(rocsolver_cgetrs)                 \
  __macro(rocsolver_dgetrs)                 \
  __macro(rocsolver_sgetrs)                 \
  __macro(rocsolver_zgetrs)                 \
  __macro(rocsolver_cgetrf_batched)         \
  __macro(rocsolver_dgetrf_batched)         \
  __macro(rocsolver_sgetrf_batched)         \
  __macro(rocsolver_zgetrf_batched)         \
  __macro(rocsolver_cgetrs_batched)         \
  __macro(rocsolver_dgetrs_batched)         \
  __macro(rocsolver_sgetrs_batched)         \
  __macro(rocsolver_zgetrs_batched)         \
  __macro(rocsolver_cgetri_batched)         \
  __macro(rocsolver_dgetri_batched)         \
  __macro(rocsolver_sgetri_batched)         \
  __macro(rocsolver_zgetri_batched)         \
  __macro(rocsolver_cpotrf)         	    \
  __macro(rocsolver_dpotrf)                 \
  __macro(rocsolver_spotrf)                 \
  __macro(rocsolver_zpotrf)                 \
  __macro(rocsolver_cpotrf_batched)         \
  __macro(rocsolver_dpotrf_batched)         \
  __macro(rocsolver_spotrf_batched)         \
  __macro(rocsolver_zpotrf_batched)         \
  __macro(rocsolver_cgeqrf)                 \
  __macro(rocsolver_dgeqrf)                 \
  __macro(rocsolver_sgeqrf)                 \
  __macro(rocsolver_zgeqrf)                 \
  __macro(rocsolver_cunmqr)                 \
  __macro(rocsolver_zunmqr)                 \
  __macro(rocsolver_cungqr)                 \
  __macro(rocsolver_zungqr)
// clang-format on

FOREACH_ROCSOLVER_API(ROCSOLVER_API_WRAPPER)

#undef FOREACH_ROCSOLVER_API
#undef ROCSOLVER_API_WRAPPER

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCSOLVER_WRAPPER_H_
