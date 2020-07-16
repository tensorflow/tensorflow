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

#include "tensorflow/compiler/mlir/init_mlir.h"

#include "tensorflow/core/platform/init_main.h"

namespace tensorflow {

InitMlir::InitMlir(int *argc, char ***argv) : init_llvm_(*argc, *argv) {
  constexpr char kSeparator[] = "--";

  // Find index of separator between two sets of flags.
  int pass_remainder = 1;
  bool split = false;
  for (int i = 0; i < *argc; ++i) {
    if (llvm::StringRef((*argv)[i]) == kSeparator) {
      pass_remainder = i;
      *argc -= (i + 1);
      split = true;
      break;
    }
  }

  tensorflow::port::InitMain((*argv)[0], &pass_remainder, argv);
  if (split) {
    *argc += pass_remainder;
    (*argv)[1] = (*argv)[0];
    ++*argv;
  }
}

}  // namespace tensorflow
