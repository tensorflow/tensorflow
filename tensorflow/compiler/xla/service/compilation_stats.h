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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_STATS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_STATS_H_

#include <memory>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace xla {

// This class is used to collect information about HLO passes and print some
// statistics at the end of compilation. From HloPassPipeline, we call StartPass
// before the execution of a pass, and EndPass after. Currently, we only collect
// timing information and how many times each pass was run. In the future, we
// can add more things, such as the size of the HLO graph after each pass.
class CompilationStats {
 public:
  virtual ~CompilationStats() = default;

  static std::unique_ptr<CompilationStats> MakeNoopStats();

  static std::unique_ptr<CompilationStats> MakeStats();

  virtual void StartPass(absl::string_view pass_name) = 0;

  virtual void EndPass(absl::string_view pass_name) = 0;

  virtual void CompilationReport() = 0;

  virtual int GetPassesSize() = 0;

  virtual void RecordPassError(absl::string_view pass_name,
                               absl::string_view err) = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_STATS_H_
