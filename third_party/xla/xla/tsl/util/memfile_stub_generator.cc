/* Copyright 2026 The OpenXLA Authors.

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
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

int main(int argc, char** argv) {
  std::string generated_file;
  std::string embed_dir;
  std::string embed_name;
  std::string dirname;
  std::vector<tsl::Flag> flags = {
      tsl::Flag("generated_file", &generated_file,
                "Required. Path to the file to write."),
      tsl::Flag("embed_dir", &embed_dir,
                "Required. Directory of the resulting memfiles."),
      tsl::Flag("embed_name", &embed_name,
                "Required. Name of the embedded data."),
      tsl::Flag("dirname", &dirname,
                "Required. Name of the directory we are in."),
  };

  QCHECK(tsl::Flags::Parse(&argc, argv, flags));

  tsl::port::InitMain("", &argc, &argv);

  QCHECK(!generated_file.empty()) << "--generated_file is required!";
  QCHECK(!embed_dir.empty()) << "--embed_dir is required!";
  QCHECK(!embed_name.empty()) << "--embed_name is required!";
  QCHECK(!dirname.empty()) << "--dirname is required!";

  std::string contents =
      absl::StrFormat(R"(#include "%s/%s.h"

#include "xla/tsl/util/memfile_builtin.h"

REGISTER_BUILTIN_FILES_WITH_DIRECTORY(%s, "%s");
)",
                      dirname, embed_name, embed_name, embed_dir);

  QCHECK_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), generated_file, contents));
}
