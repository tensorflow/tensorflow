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
#include "tensorflow/lite/tools/evaluation/tasks/task_executor.h"

#include "absl/types/optional.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace evaluation {
absl::optional<EvaluationStageMetrics> TaskExecutor::Run(int* argc,
                                                         char* argv[]) {
  auto flag_list = GetFlags();
  auto delegate_flags = delegate_providers_.GetFlags();

  flag_list.insert(flag_list.end(), delegate_flags.begin(),
                   delegate_flags.end());
  bool parse_result =
      tflite::Flags::Parse(argc, const_cast<const char**>(argv), flag_list);
  if (!parse_result) {
    std::string usage = Flags::Usage(argv[0], flag_list);
    TFLITE_LOG(ERROR) << usage;
    return absl::nullopt;
  }

  std::string unconsumed_args =
      Flags::ArgsToString(*argc, const_cast<const char**>(argv));
  if (!unconsumed_args.empty()) {
    TFLITE_LOG(WARN) << "Unconsumed cmdline flags: " << unconsumed_args;
  }

  return RunImpl();
}
}  // namespace evaluation
}  // namespace tflite
