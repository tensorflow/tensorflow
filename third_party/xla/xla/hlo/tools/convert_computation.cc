/* Copyright 2017 The OpenXLA Authors.

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

// Usage: convert_computation <txt2bin|bin2txt> serialized_computation_proto
//
// bin2txt spits out the result to stdout. txt2bin modifies the file in place.
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "tsl/platform/status.h"
#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdio.h>

#include <string>

#include "xla/service/hlo.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace tools {

void RealMain(const std::string& mode, const std::string& path) {
  HloSnapshot module;
  tsl::Env* env = tsl::Env::Default();
  if (mode == "txt2bin") {
    TF_CHECK_OK(tsl::ReadTextProto(env, path, &module));
    TF_CHECK_OK(tsl::WriteBinaryProto(env, path, module));
  } else if (mode == "bin2txt") {
    TF_CHECK_OK(tsl::ReadBinaryProto(env, path, &module));
    std::string out;
    tsl::protobuf::TextFormat::PrintToString(module, &out);
    fprintf(stdout, "%s", out.c_str());
  } else {
    LOG(QFATAL) << "unknown mode for computation conversion: " << mode;
  }
}

}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);

  QCHECK_EQ(argc, 3) << "usage: " << argv[0] << " <txt2bin|bin2txt> <path>";
  xla::tools::RealMain(argv[1], argv[2]);
  return 0;
}
