/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_PROFILING_SESSION_H_
#define XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_PROFILING_SESSION_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/profiler/subprocess/subprocess_registry.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {

// Creates a profiler for the specified subprocess. The subprocess must have a
// ProfilerService gRPC server listening on the address specified in
// `subprocess_info`.
absl::StatusOr<std::unique_ptr<tsl::profiler::ProfilerInterface>>
CreateSubprocessProfilingSession(const SubprocessInfo& subprocess_info,
                                 const tensorflow::ProfileOptions& options);

}  // namespace subprocess
}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_PROFILING_SESSION_H_
