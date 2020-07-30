/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_

#include "rocm/include/roctracer/roctracer.h"
#include "rocm/include/roctracer/roctracer_hcc.h"
#include "rocm/include/roctracer/roctracer_hip.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace tensorflow {
namespace wrap {

namespace CachedDsoLoader = stream_executor::internal::CachedDsoLoader;

#ifdef PLATFORM_GOOGLE

#define ROCTRACER_API_WRAPPER(__name)                        \
  template <typename... Args>                                \
  auto __name()(Args... args)->decltype(::__name(args...)) { \
    return ::__name(args...);                                \
  }

#else

#define ROCTRACER_API_WRAPPER(__name)                                        \
  template <typename... Args>                                                \
  auto __name(Args... args)->decltype(::__name(args...)) {                   \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;             \
    static FuncPtrT loaded = []() -> FuncPtrT {                              \
      static const char* kName = #__name;                                    \
      void* f;                                                               \
      auto s = Env::Default()->GetSymbolFromLibrary(                         \
          CachedDsoLoader::GetRoctracerDsoHandle().ValueOrDie(), kName, &f); \
      CHECK(s.ok()) << "could not find " << kName                            \
                    << " in roctracer DSO; dlerror: " << s.error_message();  \
      return reinterpret_cast<FuncPtrT>(f);                                  \
    }();                                                                     \
    return loaded(args...);                                                  \
  }

#endif

// clang-format off
#define FOREACH_ROCTRACER_API(__macro)			\
  __macro(roctracer_default_pool_expl)			\
  __macro(roctracer_disable_domain_activity)		\
  __macro(roctracer_disable_domain_callback)		\
  __macro(roctracer_disable_op_activity)		\
  __macro(roctracer_disable_op_callback)		\
  __macro(roctracer_enable_domain_activity_expl)	\
  __macro(roctracer_enable_domain_callback)		\
  __macro(roctracer_enable_op_activity)			\
  __macro(roctracer_enable_op_callback)			\
  __macro(roctracer_error_string)			\
  __macro(roctracer_flush_activity_expl)		\
  __macro(roctracer_get_timestamp)			\
  __macro(roctracer_op_string)				\
  __macro(roctracer_open_pool_expl)			\
  __macro(roctracer_set_properties)

// clang-format on

FOREACH_ROCTRACER_API(ROCTRACER_API_WRAPPER)

#undef FOREACH_ROCTRACER_API
#undef ROCTRACER_API_WRAPPER

}  // namespace wrap
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_
