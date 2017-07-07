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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_UTILS_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/profiler/internal/tfprof_options.h"

namespace tensorflow {
namespace tfprof {
string FormatNumber(int64 n);

string FormatTime(int64 micros);

string FormatMemory(int64 bytes);

string FormatShapes(const std::vector<int64>& shapes);

tensorflow::Status ParseCmdLine(const string& line, string* cmd,
                                tensorflow::tfprof::Options* opts);

string StringReplace(const string& str, const string& oldsub,
                     const string& newsub);

template <typename T>
Status ReadProtoFile(Env* env, const string& fname, T* proto,
                     bool binary_first) {
  string out;
  Status s = ReadFileToString(env, fname, &out);
  if (!s.ok()) return s;

  if (binary_first) {
    if (ReadBinaryProto(tensorflow::Env::Default(), fname, proto).ok()) {
      return Status();
    } else if (protobuf::TextFormat::ParseFromString(out, proto)) {
      return Status();
    }
  } else {
    if (protobuf::TextFormat::ParseFromString(out, proto)) {
      return Status();
    } else if (ReadBinaryProto(tensorflow::Env::Default(), fname, proto).ok()) {
      return Status();
    }
  }
  return errors::InvalidArgument("Cannot parse proto file.");
}

void PrintHelp();

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_UTILS_H_
