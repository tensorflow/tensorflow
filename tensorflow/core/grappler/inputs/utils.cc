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

#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/platform/env.h"

#include <vector>

namespace tensorflow {
namespace grappler {

bool FilesExist(const std::vector<string>& files, std::vector<Status>* status) {
  return Env::Default()->FilesExist(files, status);
}

bool FilesExist(const std::set<string>& files) {
  return FilesExist(std::vector<string>(files.begin(), files.end()), nullptr);
}

}  // End namespace grappler
}  // end namespace tensorflow
