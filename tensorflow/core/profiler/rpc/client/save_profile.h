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

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_SAVE_PROFILE_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_SAVE_PROFILE_H_

#include <ostream>
#include <string>

#include "xla/tsl/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::GetCurrentTimeStampAsString;     // NOLINT
using tsl::profiler::GetTensorBoardProfilePluginDir;  // NOLINT
using tsl::profiler::SaveGzippedToolData;             // NOLINT
using tsl::profiler::SaveProfile;                     // NOLINT
using tsl::profiler::SaveXSpace;                      // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_SAVE_PROFILE_H_
