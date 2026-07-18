
/* Copyright 2021 The OpenXLA Authors.

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

// This file wraps roctracer API calls with dso loader so that we don't need to
// have explicit linking to libroctracer. All TF hipsarse API usage should route
// through this wrapper.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_

#include "rocm/include/rocprofiler-sdk/buffer.h"  // IWYU pragma: export
#include "rocm/include/rocprofiler-sdk/buffer_tracing.h"  // IWYU pragma: export
#include "rocm/include/rocprofiler-sdk/callback_tracing.h"  // IWYU pragma: export
#include "rocm/include/rocprofiler-sdk/cxx/name_info.hpp"  // IWYU pragma: export
#include "rocm/include/rocprofiler-sdk/external_correlation.h"  // IWYU pragma: export
#include "rocm/include/rocprofiler-sdk/fwd.h"  // IWYU pragma: export
#include "rocm/include/rocprofiler-sdk/internal_threading.h"  // IWYU pragma: export
#include "rocm/include/rocprofiler-sdk/registration.h"  // IWYU pragma: export
#include "rocm/include/rocprofiler-sdk/rocprofiler.h"  // IWYU pragma: export

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_
