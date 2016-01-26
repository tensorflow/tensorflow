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

#include "tensorflow/cc/ops/cc_op_gen.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

void PrintAllCCOps(const std::string& dot_h, const std::string& dot_cc,
                   bool include_internal) {
  OpList ops;
  OpRegistry::Global()->Export(include_internal, &ops);
  WriteCCOps(ops, dot_h, dot_cc);
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 4) {
    fprintf(stderr,
            "Usage: %s out.h out.cc include_internal\n"
            "  include_internal: 1 means include internal ops\n",
            argv[0]);
    exit(1);
  }

  bool include_internal = tensorflow::StringPiece("1") == argv[3];
  tensorflow::PrintAllCCOps(argv[1], argv[2], include_internal);
  return 0;
}
