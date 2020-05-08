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

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"

namespace tensorflow {
namespace profiler {

string GetCurrentTimeStampAsString();

// Returns the profile plugin directory given a logdir to TensorBoard.
string GetTensorBoardProfilePluginDir(const string& logdir);

// Saves all profiling tool data in a profile to a TensorBoard log directory
// with the given run name. This writes user-facing log messages to `os`.
// Note: this function creates a directory even when all fields in
// ProfileResponse are unset/empty.
Status SaveTensorboardProfile(const string& logdir, const string& run,
                              const string& host,
                              const ProfileResponse& response,
                              std::ostream* os);

// Gzip the data and save to the specified filepath.
Status SaveGzippedToolDataToTensorboardProfile(const string& logdir,
                                               const string& run,
                                               const string& host,
                                               const string& tool_name,
                                               const string& data);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_SAVE_PROFILE_H_
