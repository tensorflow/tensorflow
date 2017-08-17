/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
#define THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_

#include <unordered_map>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A class that stores all the FunctionLibraryRuntime objects, one per device.
// This class is not thread safe.
class ProcessFunctionLibraryRuntime {
 public:
  // Creates FunctionLibraryRuntime objects for each device in the provided
  // DeviceMgr. Caller needs to make sure that device_mgr and lib_def outlive
  // this object.
  ProcessFunctionLibraryRuntime(const DeviceMgr* device_mgr, Env* env,
                                int graph_def_version,
                                const FunctionLibraryDefinition* lib_def,
                                const OptimizerOptions& optimizer_options);

  // Returns the FunctionLibraryRuntime for the corresponding device_name.
  FunctionLibraryRuntime* GetFLR(const string& device_name);

 private:
  std::unordered_map<string, std::unique_ptr<FunctionLibraryRuntime>> flr_map_;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
