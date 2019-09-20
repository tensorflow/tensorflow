/*
 * Copyright 2017 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <functional>
#include <limits>
#include <string>
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#ifndef FLATC_H_
#  define FLATC_H_

namespace flatbuffers {

class FlatCompiler {
 public:
  // Output generator for the various programming languages and formats we
  // support.
  struct Generator {
    typedef bool (*GenerateFn)(const flatbuffers::Parser &parser,
                               const std::string &path,
                               const std::string &file_name);
    typedef std::string (*MakeRuleFn)(const flatbuffers::Parser &parser,
                                      const std::string &path,
                                      const std::string &file_name);

    GenerateFn generate;
    const char *generator_opt_short;
    const char *generator_opt_long;
    const char *lang_name;
    bool schema_only;
    GenerateFn generateGRPC;
    flatbuffers::IDLOptions::Language lang;
    const char *generator_help;
    MakeRuleFn make_rule;
  };

  typedef void (*WarnFn)(const FlatCompiler *flatc, const std::string &warn,
                         bool show_exe_name);

  typedef void (*ErrorFn)(const FlatCompiler *flatc, const std::string &err,
                          bool usage, bool show_exe_name);

  // Parameters required to initialize the FlatCompiler.
  struct InitParams {
    InitParams()
        : generators(nullptr),
          num_generators(0),
          warn_fn(nullptr),
          error_fn(nullptr) {}

    const Generator *generators;
    size_t num_generators;
    WarnFn warn_fn;
    ErrorFn error_fn;
  };

  explicit FlatCompiler(const InitParams &params) : params_(params) {}

  int Compile(int argc, const char **argv);

  std::string GetUsageString(const char *program_name) const;

 private:
  void ParseFile(flatbuffers::Parser &parser, const std::string &filename,
                 const std::string &contents,
                 std::vector<const char *> &include_directories) const;

  void LoadBinarySchema(Parser &parser, const std::string &filename,
                        const std::string &contents);

  void Warn(const std::string &warn, bool show_exe_name = true) const;

  void Error(const std::string &err, bool usage = true,
             bool show_exe_name = true) const;

  InitParams params_;
};

}  // namespace flatbuffers

#endif  // FLATC_H_
