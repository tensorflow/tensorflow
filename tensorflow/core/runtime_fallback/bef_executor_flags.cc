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

#include "tensorflow/core/runtime_fallback/bef_executor_flags.h"

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

ABSL_FLAG(std::string, input_filename, tfrt::kDefaultInputFilename,
          "mlir input filename (default '-' for stdin)");
ABSL_FLAG(std::string, shared_libs, "",
          "comma-separated list of dynamic libraries with ops");
ABSL_FLAG(std::string, functions, "",
          "comma-separated list of mlir functions to run");
ABSL_FLAG(std::string, test_init_function, "",
          "init function that will be invoked as part of "
          "initialization, before invoking any other MLIR functions even if it "
          "is not specified in --functions flag.");
ABSL_FLAG(std::string, work_queue_type, "s",
          "type of work queue (s(default), mstd, ...)");
ABSL_FLAG(tfrt::HostAllocatorTypeWrapper, host_allocator_type,
          {tfrt::HostAllocatorType::kLeakCheckMalloc},
          "type of host allocator (malloc, profiled_allocator, "
          "leak_check_allocator(default))");

namespace tfrt {

const char kDefaultInputFilename[] = "-";

// AbslParseFlag/AbslUnparseFlag need to be in tfrt namespace for ADL to work.

bool AbslParseFlag(absl::string_view text,
                   tfrt::HostAllocatorTypeWrapper* host_allocator_type,
                   std::string* error) {
  if (text == "malloc") {
    *host_allocator_type = {tfrt::HostAllocatorType::kMalloc};
    return true;
  }
  if (text == "profiled_allocator") {
    *host_allocator_type = {tfrt::HostAllocatorType::kProfiledMalloc};
    return true;
  }
  if (text == "leak_check_allocator") {
    *host_allocator_type = {tfrt::HostAllocatorType::kLeakCheckMalloc};
    return true;
  }
  *error = "Unknown value for tfrt::HostAllocatorType";
  return false;
}

std::string AbslUnparseFlag(
    tfrt::HostAllocatorTypeWrapper host_allocator_type) {
  switch (host_allocator_type) {
    case tfrt::HostAllocatorType::kMalloc:
      return "malloc";
    case tfrt::HostAllocatorType::kTestFixedSizeMalloc:
      return "test_fixed_size_1k";
    case tfrt::HostAllocatorType::kProfiledMalloc:
      return "profiled_allocator";
    case tfrt::HostAllocatorType::kLeakCheckMalloc:
      return "leak_check_allocator";
  }
}
}  // namespace tfrt
