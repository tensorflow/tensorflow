/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_DUMP_TPU_PROFILE_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_DUMP_TPU_PROFILE_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/grpc_services.h"

namespace tensorflow {

namespace profiler {
namespace client {

// Dumps all profiling tool data in a TPU profile to a TensorBoard log directory
// with the given run name. This writes user-facing log messages to `os`.
// The following tools are supported:
//   - Trace viewer
//   - Op profile
//   - Input pipeline analyzer
//   - Overview page
// Note: this function creates a directory even when all fields in
// ProfileResponse are unset/empty.
Status WriteTensorboardTPUProfile(const string& logdir, const string& run,
                                  const string& host,
                                  const ProfileResponse& response,
                                  std::ostream* os);

}  // namespace client
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_DUMP_TPU_PROFILE_H_
