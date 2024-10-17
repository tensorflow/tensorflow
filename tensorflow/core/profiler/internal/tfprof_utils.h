/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_UTILS_H_

#include <string>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/tfprof_options.h"

namespace tensorflow {
namespace tfprof {
string FormatNumber(int64_t n);

string FormatTime(int64_t micros);

string FormatMemory(int64_t bytes);

string FormatShapes(const std::vector<int64_t>& shapes);

absl::Status ParseCmdLine(const string& line, string* cmd,
                          tensorflow::tfprof::Options* opts);

string StringReplace(const string& str, const string& oldsub,
                     const string& newsub);

template <typename T>
absl::Status ReadProtoFile(Env* env, const string& fname, T* proto,
                           bool binary_first) {
  string out;
  absl::Status s = ReadFileToString(env, fname, &out);
  if (!s.ok()) return s;

  if (binary_first) {
    if (ReadBinaryProto(tensorflow::Env::Default(), fname, proto).ok()) {
      return absl::Status();
    } else if (protobuf::TextFormat::ParseFromString(out, proto)) {
      return absl::Status();
    }
  } else {
    if (protobuf::TextFormat::ParseFromString(out, proto)) {
      return absl::Status();
    } else if (ReadBinaryProto(tensorflow::Env::Default(), fname, proto).ok()) {
      return absl::Status();
    }
  }
  return errors::InvalidArgument("Cannot parse proto file.");
}

void PrintHelp();

// Generate helper message based on the command and options.
string QueryDoc(const string& cmd, const Options& opts);

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_UTILS_H_
