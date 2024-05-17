/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_BEF_EXECUTOR_FLAGS_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_BEF_EXECUTOR_FLAGS_H_

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "tfrt/bef_executor_driver/bef_executor_driver.h"  // from @tf_runtime

namespace tfrt {
ABSL_CONST_INIT extern const char kDefaultInputFilename[];

struct HostAllocatorTypeWrapper {
  HostAllocatorTypeWrapper(HostAllocatorType type) : type(type) {}
  operator HostAllocatorType() { return type; }
  HostAllocatorType type;
};

}  // namespace tfrt

ABSL_DECLARE_FLAG(std::string, input_filename);
ABSL_DECLARE_FLAG(std::string, shared_libs);
ABSL_DECLARE_FLAG(std::string, functions);
ABSL_DECLARE_FLAG(std::string, test_init_function);
ABSL_DECLARE_FLAG(std::string, work_queue_type);
ABSL_DECLARE_FLAG(tfrt::HostAllocatorTypeWrapper, host_allocator_type);

namespace tfrt {

bool AbslParseFlag(absl::string_view text,
                   tfrt::HostAllocatorTypeWrapper* host_allocator_type,
                   std::string* error);

std::string AbslUnparseFlag(tfrt::HostAllocatorTypeWrapper host_allocator_type);

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_BEF_EXECUTOR_FLAGS_H_
