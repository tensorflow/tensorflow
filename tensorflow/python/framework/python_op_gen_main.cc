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

#include "tensorflow/python/framework/python_op_gen.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

void PrintAllPythonOps(const char* hidden, bool require_shapes) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);
  PrintPythonOps(ops, hidden, require_shapes);
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc == 2) {
    tensorflow::PrintAllPythonOps("", std::string(argv[1]) == "1");
  } else if (argc == 3) {
    tensorflow::PrintAllPythonOps(argv[1], std::string(argv[2]) == "1");
  } else {
    return -1;
  }
  return 0;
}
