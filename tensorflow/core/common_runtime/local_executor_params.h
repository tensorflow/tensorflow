/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_LOCAL_EXECUTOR_PARAMS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_LOCAL_EXECUTOR_PARAMS_H_

#include <functional>
#include <memory>

namespace tensorflow {

class Device;
class StepStatsCollector;
class SessionMetadata;
class FunctionLibraryRuntime;
class NodeProperties;
class OpKernel;
class Status;

// LocalExecutorParams provides arguments that will be shared by all invocations
// of an executor. We expect that different contexts would provide different
// implementations (e.g. local versus distributed).
struct LocalExecutorParams {
  Device* device;

  const SessionMetadata* session_metadata = nullptr;

  // The library runtime support.
  FunctionLibraryRuntime* function_library = nullptr;

  // create_kernel returns an instance of op kernel based on NodeDef.
  // delete_kernel is called for every kernel used by the executor
  // when the executor is deleted.
  std::function<Status(const std::shared_ptr<const NodeProperties>&,
                       OpKernel**)>
      create_kernel;
  std::function<void(OpKernel*)> delete_kernel;

  // Whether control flow nodes are allowed to be executed synchronously.
  bool allow_control_flow_sync_execution = false;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_LOCAL_EXECUTOR_PARAMS_H_
