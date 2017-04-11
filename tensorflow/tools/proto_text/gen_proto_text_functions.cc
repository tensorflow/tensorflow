/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

static const char kTensorFlowHeaderPrefix[] = "";

static const char kPlaceholderFile[] =
    "tensorflow/tools/proto_text/placeholder.txt";

bool IsPlaceholderFile(const char* s) {
  string ph(kPlaceholderFile);
  string str(s);
  return str.size() >= strlen(kPlaceholderFile) &&
         ph == str.substr(str.size() - ph.size());
}

}  // namespace

// Main program to take input protos and write output pb_text source files that
// contain generated proto text input and output functions.
//
// Main expects:
// - First argument is output path
// - Second argument is the relative path of the protos to the root. E.g.,
//   for protos built by a rule in tensorflow/core, this will be
//   tensorflow/core.
// - Then any number of source proto file names, plus one source name must be
//   placeholder.txt from this gen tool's package.  placeholder.txt is
//   ignored for proto resolution, but is used to determine the root at which
//   the build tool has placed the source proto files.
//
// Note that this code doesn't use tensorflow's command line parsing, because of
// circular dependencies between libraries if that were done.
//
// This is meant to be invoked by a genrule. See BUILD for more information.
int MainImpl(int argc, char** argv) {
  if (argc < 4) {
    LOG(ERROR) << "Pass output path, relative path, and at least proto file";
    return -1;
  }

  const string output_root = argv[1];
  const string output_relative_path = kTensorFlowHeaderPrefix + string(argv[2]);

  string src_relative_path;
  bool has_placeholder = false;
  for (int i = 3; i < argc; ++i) {
    if (IsPlaceholderFile(argv[i])) {
      const string s(argv[i]);
      src_relative_path = s.substr(0, s.size() - strlen(kPlaceholderFile));
      has_placeholder = true;
    }
  }
  if (!has_placeholder) {
    LOG(ERROR) << kPlaceholderFile << " must be passed";
    return -1;
  }

  tensorflow::protobuf::compiler::DiskSourceTree source_tree;

  source_tree.MapPath("", src_relative_path.empty() ? "." : src_relative_path);
  CrashOnErrorCollector crash_on_error;
  tensorflow::protobuf::compiler::Importer importer(&source_tree,
                                                    &crash_on_error);

  for (int i = 3; i < argc; i++) {
    if (IsPlaceholderFile(argv[i])) continue;
    const string proto_path = string(argv[i]).substr(src_relative_path.size());

    const tensorflow::protobuf::FileDescriptor* fd =
        importer.Import(proto_path);

    const int index = proto_path.find_last_of(".");
    string proto_path_no_suffix = proto_path.substr(0, index);

    proto_path_no_suffix =
        proto_path_no_suffix.substr(output_relative_path.size());

    const auto code =
        tensorflow::GetProtoTextFunctionCode(*fd, kTensorFlowHeaderPrefix);

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

      const string path = output_root + "/" + proto_path_no_suffix + suffix;
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
