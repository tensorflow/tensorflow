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

#include "tensorflow/core/kernels/ops_testutil.h"

#include <vector>

namespace tensorflow {
namespace test {

NodeDef Node(const string& name, const string& op,
             const std::vector<string>& inputs) {
  NodeDef def;
  def.set_name(name);
  def.set_op(op);
  for (const string& s : inputs) {
    def.add_input(s);
  }
  return def;
}

}  // namespace test
}  // namespace tensorflow
