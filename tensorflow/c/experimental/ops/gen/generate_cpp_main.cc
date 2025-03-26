/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/c/experimental/ops/gen/common/path_config.h"
#include "tensorflow/c/experimental/ops/gen/cpp/cpp_generator.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/cpp_config.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::string;

namespace generator = tensorflow::generator;

namespace {
class MainConfig {
 public:
  void InitMain(int* argc, char*** argv) {
    std::vector<tensorflow::Flag> flags = Flags();

    // Parse known flags
    string usage = tensorflow::Flags::Usage(
        absl::StrCat(*argv[0], " Op1 [Op2 ...]"), flags);
    QCHECK(tensorflow::Flags::Parse(argc, *argv, flags)) << usage;  // Crash OK

    // Initialize any TensorFlow support, parsing boilerplate flags (e.g. logs)
    tensorflow::port::InitMain(usage.c_str(), argc, argv);

    // Validate flags
    if (help_) {
      LOG(QFATAL) << usage;  // Crash OK
    }

    QCHECK(!source_dir_.empty()) << usage;  // Crash OK
    QCHECK(!output_dir_.empty()) << usage;  // Crash OK
    QCHECK(!category_.empty()) << usage;    // Crash OK

    // Remaining arguments (i.e. the positional args) are the requested Op names
    op_names_.assign((*argv) + 1, (*argv) + (*argc));
  }

  generator::cpp::CppConfig CppConfig() {
    return generator::cpp::CppConfig(category_);
  }

  generator::PathConfig PathConfig() {
    return generator::PathConfig(output_dir_, source_dir_, api_dirs_,
                                 op_names_);
  }

 private:
  std::vector<tensorflow::Flag> Flags() {
    return {
        tensorflow::Flag("help", &help_, "Print this help message."),
        tensorflow::Flag("category", &category_,
                         "Category for generated ops (e.g. 'math', 'array')."),
        tensorflow::Flag(
            "namespace", &name_space_,
            "Compact C++ namespace, default is 'tensorflow::ops'."),
        tensorflow::Flag(
            "output_dir", &output_dir_,
            "Directory into which output files will be generated."),
        tensorflow::Flag(
            "source_dir", &source_dir_,
            "The tensorflow root directory, e.g. 'tensorflow/' for "
            "in-source include paths. Any path underneath the "
            "tensorflow root is also accepted."),
        tensorflow::Flag(
            "api_dirs", &api_dirs_,
            "Comma-separated list of directories containing API definitions.")};
  }

  bool help_ = false;
  string category_;
  string name_space_;
  string output_dir_;
  string source_dir_;
  string api_dirs_;
  std::vector<string> op_names_;
};

}  // namespace

int main(int argc, char* argv[]) {
  MainConfig config;
  config.InitMain(&argc, &argv);
  generator::CppGenerator generator(config.CppConfig(), config.PathConfig());
  generator.WriteHeaderFile();
  generator.WriteSourceFile();
  return 0;
}
