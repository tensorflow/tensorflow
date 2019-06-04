/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

GTEST_API_ int main(int real_argc, char** real_argv) {
  std::vector<tensorflow::Flag> flag_list;
  tensorflow::AppendMarkForCompilationPassFlags(&flag_list);
  auto usage = tensorflow::Flags::Usage(real_argv[0], flag_list);

  std::vector<char*> args;

  args.reserve(real_argc + 1);
  for (int i = 0; i < real_argc; i++) {
    args.push_back(real_argv[i]);
  }

  struct FreeDeleter {
    void operator()(char* ptr) { free(ptr); }
  };

  std::unique_ptr<char, FreeDeleter> enable_global_jit_arg(
      strdup("--tf_xla_cpu_global_jit=true"));
  args.push_back(enable_global_jit_arg.get());

  std::unique_ptr<char, FreeDeleter> reduce_min_cluster_size_arg(
      strdup("--tf_xla_min_cluster_size=2"));
  args.push_back(reduce_min_cluster_size_arg.get());

  int argc = args.size();

  if (!tensorflow::Flags::Parse(&argc, &args.front(), flag_list)) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }

  testing::InitGoogleTest(&argc, &args.front());
  return RUN_ALL_TESTS();
}
