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

#ifdef PLATFORM_GOOGLE

#define ROCTRACER_API_WRAPPER(xxx_name)					\
  template <typename... Args>						\
  auto xxx_name()(Args... args)->decltype(::xxx_name(args...)) {	\
    return ::xxx_name(args...);						\
  }

#else

#define ROCTRACER_API_WRAPPER(xxx_name)					       \
  template <typename... Args>						       \
  auto xxx_name(Args... args)->decltype(::xxx_name(args...)) {		       \
    using FuncPtrT = std::add_pointer<decltype(::xxx_name)>::type;	       \
    static FuncPtrT loaded = []() -> FuncPtrT {				       \
      static const char* kName = #xxx_name;				       \
      void* f;								       \
      auto s = Env::Default()->GetSymbolFromLibrary(			       \
          stream_executor::internal::CachedDsoLoader::GetRoctracerDsoHandle(   \
  ).ValueOrDie(), kName, &f);						       \
      CHECK(s.ok()) << "could not find " << kName			       \
                    << " in roctracer DSO; dlerror: " << s.error_message();    \
      return reinterpret_cast<FuncPtrT>(f);				       \
    }();								       \
    return loaded(args...);						       \
  }

#endif

// clang-format off
#define FOREACH_ROCTRACER_API(xxx_macro)		\
  xxx_macro(roctracer_default_pool_expl)		\
  xxx_macro(roctracer_disable_domain_activity)		\
  xxx_macro(roctracer_disable_domain_callback)		\
  xxx_macro(roctracer_disable_op_activity)		\
  xxx_macro(roctracer_disable_op_callback)		\
  xxx_macro(roctracer_enable_domain_activity_expl)	\
  xxx_macro(roctracer_enable_domain_callback)		\
  xxx_macro(roctracer_enable_op_activity)		\
  xxx_macro(roctracer_enable_op_callback)		\
  xxx_macro(roctracer_error_string)			\
  xxx_macro(roctracer_flush_activity_expl)		\
  xxx_macro(roctracer_get_timestamp)			\
  xxx_macro(roctracer_op_string)			\
  xxx_macro(roctracer_open_pool_expl)			\
  xxx_macro(roctracer_set_properties)

// clang-format on

FOREACH_ROCTRACER_API(ROCTRACER_API_WRAPPER)

#undef FOREACH_ROCTRACER_API
#undef ROCTRACER_API_WRAPPER

}  // namespace wrap
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_
