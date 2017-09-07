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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/java/src/gen/cc/op_generator.h"

namespace tensorflow {
namespace op_gen {

const char kUsageHeader[] =
    "\n\nGenerator of operation wrappers in Java.\n\n"
    "This executable generates wrappers for all registered operations it has "
    "been compiled with. A wrapper exposes an intuitive and strongly-typed\n"
    "interface for building its underlying operation and linking it into a "
    "graph.\n\n"
    "Operation wrappers are generated under the path specified by the "
    "'--output_dir' argument. This path can be absolute or relative to the\n"
    "current working directory and will be created if it does not exists.\n\n"
    "The '--lib_name' argument is used to classify the set of operations. If "
    "the chosen name contains more than one word, it must be provided in \n"
    "snake_case. This value is declined into other meaningful names, such as "
    "the group and package of the generated operations. For example,\n"
    "'--lib_name=my_lib' generates the operations under the "
    "'org.tensorflow.op.mylib' package and add them to the 'myLib()' operator\n"
    "group.\n\n"
    "Note that the operator group assigned to the generated wrappers is just "
    "an annotation tag at this stage. Operations will not be available "
    "through\n"
    "the 'org.tensorflow.op.Ops' API as a group until the generated classes "
    "are compiled using an appropriate annotation processor.\n\n"
    "Finally, the '--base_package' overrides the default parent package "
    "under which the generated subpackage and classes are to be located.\n\n";

}  // namespace op_gen
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::string lib_name;
  tensorflow::string output_dir;
  tensorflow::string base_package = "org.tensorflow.op";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("output_dir", &output_dir,
                       "Root directory into which output files are generated"),
      tensorflow::Flag(
          "lib_name", &lib_name,
          "A name, in snake_case, used to classify this set of operations"),
      tensorflow::Flag(
          "base_package", &base_package,
          "Package parent to the generated subpackage and classes")};
  tensorflow::string usage = tensorflow::op_gen::kUsageHeader;
  usage += tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(parsed_flags_ok && !lib_name.empty() && !output_dir.empty()) << usage;

  tensorflow::OpGenerator generator;
  tensorflow::OpList ops;
  tensorflow::OpRegistry::Global()->Export(true, &ops);
  tensorflow::Status status =
      generator.Run(ops, lib_name, base_package, output_dir);
  TF_QCHECK_OK(status);

  return 0;
}
