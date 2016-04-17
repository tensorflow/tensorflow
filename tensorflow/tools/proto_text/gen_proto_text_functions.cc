/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include <stdio.h>
#include <set>

#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tools/proto_text/gen_proto_text_functions_lib.h"

namespace tensorflow {

namespace {
class CrashOnErrorCollector
    : public tensorflow::protobuf::compiler::MultiFileErrorCollector {
 public:
  ~CrashOnErrorCollector() override {}

  void AddError(const string& filename, int line, int column,
                const string& message) override {
    LOG(FATAL) << "Unexpected error at " << filename << "@" << line << ":"
               << column << " - " << message;
  }
};
}  // namespace

static const char kTensorflowHeaderPrefix[] = "";

// Main program to take input protos and write output pb_text source files that
// contain generated proto text input and output functions.
//
// Main expects the first argument to give the output path. This is followed by
// pairs of arguments: <proto_name_relative_to_root, proto_file_path>.
//
// Note that this code doesn't use tensorflow's command line parsing, because of
// circular dependencies between libraries if that were done.
//
// This is meant to be invoked by a genrule. See BUILD for more information.
int MainImpl(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  if (argc < 3) {
    LOG(ERROR) << "Pass output path and at least proto file";
    return -1;
  }

  const string output_root = argv[1];

  tensorflow::protobuf::compiler::DiskSourceTree source_tree;

  // This requires all protos to be relative to the directory from which the
  // genrule is invoked. If protos are generated in some other directory,
  // then they may not be found.
  source_tree.MapPath("", ".");
  CrashOnErrorCollector crash_on_error;
  tensorflow::protobuf::compiler::Importer importer(&source_tree,
                                                    &crash_on_error);

  for (int i = 2; i < argc; i++) {
    const string proto_path = argv[i];
    const tensorflow::protobuf::FileDescriptor* fd =
        importer.Import(proto_path);

    string proto_name = proto_path;
    int index = proto_name.find_last_of("/");
    if (index != string::npos) proto_name = proto_name.substr(index + 1);
    index = proto_name.find_last_of(".");
    if (index != string::npos) proto_name = proto_name.substr(0, index);

    const auto code =
        tensorflow::GetProtoTextFunctionCode(*fd, kTensorflowHeaderPrefix);

    // Three passes, one for each output file.
    for (int pass = 0; pass < 3; ++pass) {
      string suffix;
      string data;
      if (pass == 0) {
        suffix = ".pb_text.h";
        data = code.header;
      } else if (pass == 1) {
        suffix = ".pb_text-impl.h";
        data = code.header_impl;
      } else {
        suffix = ".pb_text.cc";
        data = code.cc;
      }

      const string path = output_root + "/" + proto_name + suffix;
      FILE* f = fopen(path.c_str(), "w");
      if (fwrite(data.c_str(), 1, data.size(), f) != data.size()) {
        return -1;
      }
      if (fclose(f) != 0) {
        return -1;
      }
    }
  }
  return 0;
}

}  // namespace tensorflow

int main(int argc, char** argv) { return tensorflow::MainImpl(argc, argv); }
