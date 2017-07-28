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

#include <string>

#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/java/src/gen/cc/ops/op_generator.h"

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  const std::string& output_dir = argv[1];
  const std::string& ops_lib = argv[2];

  LOG(INFO) << "Generating Java operation wrappers for \""
            << ops_lib << "\" library";

  tensorflow::OpGenerator generator(ops_lib);

  return generator.Run(output_dir + "/src/main/java/");
}
