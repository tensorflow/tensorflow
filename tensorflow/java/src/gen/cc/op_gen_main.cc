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
#include <vector>

#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/java/src/gen/cc/op_generator.h"

namespace tensorflow {
namespace op_gen {

const char kUsageHeader[] =
    "\n\nGenerator of operation wrappers in Java.\n\n"
    "This executable generates wrappers for all operations registered in the\n"
    "ops file it has been linked to (i.e. one of the /core/ops/*.o binaries).\n"
    "Generated files are output to the path provided as an argument, under\n"
    "their appropriate package and using a maven-style directory layout.\n\n";

}  // namespace op_gen
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::string ops_file;
  tensorflow::string output_dir;
  std::vector<tensorflow::Flag> flag_list = {
    tensorflow::Flag("file", &ops_file,
        "name of the ops file linked to this executable"),
    tensorflow::Flag("output", &output_dir,
        "base directory where to output generated files")
  };
  tensorflow::string usage = tensorflow::op_gen::kUsageHeader;
  usage += tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok && !ops_file.empty() && !output_dir.empty()) << usage;
  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);

  tensorflow::OpGenerator generator(tensorflow::Env::Default(), output_dir);
  tensorflow::OpList ops;
  tensorflow::OpRegistry::Global()->Export(true, &ops);
  tensorflow::Status status = generator.Run(ops_file, ops);
  TF_QCHECK_OK(status);

  return 0;
}
