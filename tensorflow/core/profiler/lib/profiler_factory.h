/* Copyright 2019 The TensorFlow Authors All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_PROFILER_FACTORY_H_
#define TENSORFLOW_CORE_PROFILER_LIB_PROFILER_FACTORY_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/macros.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

// TODO: b/323943471 - This macro should eventually be provided by Abseil.
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif

namespace tensorflow {
namespace profiler {

// A ProfilerFactory returns an instance of ProfilerInterface if ProfileOptions
// require it. Otherwise, it might return nullptr.
using ProfilerFactor ABSL_DEPRECATE_AND_INLINE() =
    tsl::profiler::ProfilerFactory;  // NOLINT

// Registers a profiler factory. Should be invoked at most once per factory.
ABSL_DEPRECATE_AND_INLINE()
inline void RegisterProfilerFactory(tsl::profiler::ProfilerFactory factory) {
  tsl::profiler::RegisterProfilerFactory(std::move(factory));
}

// Invokes all registered profiler factories with the given options, and
// returns the instantiated (non-null) profiler interfaces.
ABSL_DEPRECATE_AND_INLINE()
inline std::vector<std::unique_ptr<tsl::profiler::ProfilerInterface>>
CreateProfilers(const tensorflow::ProfileOptions& options) {
  return tsl::profiler::CreateProfilers(options);
}

// For testing only.
ABSL_DEPRECATE_AND_INLINE()
inline void ClearRegisteredProfilersForTest() {
  tsl::profiler::ClearRegisteredProfilersForTest();
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_PROFILER_FACTORY_H_
