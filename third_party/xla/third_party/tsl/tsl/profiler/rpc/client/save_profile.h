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

#ifndef TENSORFLOW_TSL_PROFILER_RPC_CLIENT_SAVE_PROFILE_H_
#define TENSORFLOW_TSL_PROFILER_RPC_CLIENT_SAVE_PROFILE_H_

#include <ostream>
#include <string>

#include "tsl/platform/status.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

std::string GetCurrentTimeStampAsString();

// Returns the profile plugin directory given a logdir to TensorBoard.
std::string GetTensorBoardProfilePluginDir(const std::string& logdir);

// Saves all profiling tool data in a profile to <repository_root>/<run>/.
// This writes user-facing log messages to `os`.
// Note: this function creates a directory even when all fields in
// ProfileResponse are unset/empty.
Status SaveProfile(const std::string& repository_root, const std::string& run,
                   const std::string& host,
                   const tensorflow::ProfileResponse& response,
                   std::ostream* os);

// Gzip the data and save to <repository_root>/<run>/.
Status SaveGzippedToolData(const std::string& repository_root,
                           const std::string& run, const std::string& host,
                           const std::string& tool_name,
                           const std::string& data);

// Save XSpace to <repository_root>/<run>/<host>_<port>.<kXPlanePb>.
Status SaveXSpace(const std::string& repository_root, const std::string& run,
                  const std::string& host,
                  const tensorflow::profiler::XSpace& xspace);

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_RPC_CLIENT_SAVE_PROFILE_H_
