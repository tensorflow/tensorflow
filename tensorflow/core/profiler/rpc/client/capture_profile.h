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
// GRPC client to perform on-demand profiling

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_CAPTURE_PROFILE_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_CAPTURE_PROFILE_H_

#include <string>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/protobuf/profiler_options.pb.h"
#include "tensorflow/tsl/profiler/protobuf/profiler_service.pb.h"
#include "tensorflow/tsl/profiler/rpc/client/capture_profile.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::ExportToTensorBoard;  // NOLINT
using tsl::profiler::Monitor;              // NOLINT
using tsl::profiler::Trace;                // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_CAPTURE_PROFILE_H_
