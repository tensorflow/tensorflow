/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/io/match.h"
#include <fnmatch.h>
#include <vector>
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {

Status GetMatchingFiles(Env* env, const string& pattern,
                        std::vector<string>* results) {
  results->clear();
  std::vector<string> all_files;
  string dir = Dirname(pattern).ToString();
  if (dir.empty()) dir = ".";
  string basename_pattern = Basename(pattern).ToString();
  Status s = env->GetChildren(dir, &all_files);
  if (!s.ok()) {
    return s;
  }
  for (const auto& f : all_files) {
    int flags = 0;
    if (fnmatch(basename_pattern.c_str(), Basename(f).ToString().c_str(),
                flags) == 0) {
      results->push_back(JoinPath(dir, f));
    }
  }
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
