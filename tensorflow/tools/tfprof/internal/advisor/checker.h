/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_CHECKER_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_CHECKER_H_

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/tools/tfprof/internal/tfprof_stats.h"

namespace tensorflow {
namespace tfprof {

static const char* const kLevel[] = {
    "NOTE",     // Good to know.
    "SUGGEST",  // Might get better.
    "WARN",     // Please do it for better.
};

class Checker {
 public:
  virtual ~Checker(){};

  virtual string name() = 0;

  std::vector<string> Run(const TFStats* stats) { return Check(stats); }

 protected:
  // Returns a vector of string, each one being an advice.
  virtual std::vector<string> Check(const TFStats* stats) = 0;
};
}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_CHECKER_H_
