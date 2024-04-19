/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_REMOTE_PROFILER_SESSION_MANAGER_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_REMOTE_PROFILER_SESSION_MANAGER_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/rpc/client/profiler_client.h"
#include "tsl/profiler/rpc/client/remote_profiler_session_manager.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::AddressResolver;               // NOLINT
using tsl::profiler::RemoteProfilerSessionManager;  // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_REMOTE_PROFILER_SESSION_MANAGER_H_
