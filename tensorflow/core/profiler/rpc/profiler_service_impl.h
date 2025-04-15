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
#ifndef TENSORFLOW_CORE_PROFILER_RPC_PROFILER_SERVICE_IMPL_H_
#define TENSORFLOW_CORE_PROFILER_RPC_PROFILER_SERVICE_IMPL_H_

#include <memory>

#include "absl/base/macros.h"
#include "xla/tsl/profiler/rpc/profiler_service_impl.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"

// TODO: b/323943471 - This macro should eventually be provided by Abseil.
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif

namespace tensorflow {
namespace profiler {

ABSL_DEPRECATE_AND_INLINE()
inline std::unique_ptr<tensorflow::grpc::ProfilerService::Service>
CreateProfilerService() {
  return tsl::profiler::CreateProfilerService();
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_PROFILER_SERVICE_IMPL_H_
