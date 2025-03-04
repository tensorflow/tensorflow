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

#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "xla/tsl/profiler/rpc/client/save_profile.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

// TODO: b/323943471 - This macro should eventually be provided by Abseil.
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif

namespace tensorflow {
namespace profiler {

ABSL_DEPRECATE_AND_INLINE()
inline std::string GetCurrentTimeStampAsString() {
  return tsl::profiler::GetCurrentTimeStampAsString();
}

ABSL_DEPRECATE_AND_INLINE()
inline std::string GetTensorBoardProfilePluginDir(const std::string& logdir) {
  return tsl::profiler::GetTensorBoardProfilePluginDir(logdir);
}

ABSL_DEPRECATE_AND_INLINE()
inline absl::Status SaveGzippedToolData(const std::string& repository_root,
                                        const std::string& run,
                                        const std::string& host,
                                        const std::string& tool_name,
                                        const std::string& data) {
  return tsl::profiler::SaveGzippedToolData(repository_root, run, host,
                                            tool_name, data);
}

ABSL_DEPRECATE_AND_INLINE()
inline absl::Status SaveProfile(const std::string& repository_root,
                                const std::string& run, const std::string& host,
                                const tensorflow::ProfileResponse& response,
                                std::ostream* os) {
  return tsl::profiler::SaveProfile(repository_root, run, host, response, os);
}

ABSL_DEPRECATE_AND_INLINE()
inline absl::Status SaveXSpace(const std::string& repository_root,
                               const std::string& run, const std::string& host,
                               const tensorflow::profiler::XSpace& xspace) {
  return tsl::profiler::SaveXSpace(repository_root, run, host, xspace);
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_SAVE_PROFILE_H_
