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

// Usage: show_signature some_binary_snapshot_proto*
//
// Shows the signature (ProgramShape) of binary snapshot proto(s) on the command
// line.
//
// some_binary_snapshot_proto is obtained by serializing the HloSnapshot from
// ServiceInterface::SnapshotComputation to disk.
//
// The output format is:
//
// file_path: computation_name :: program_shape_str

#include <stdio.h>
#include <memory>
#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace tools {

void RealMain(absl::Span<char* const> args) {
  Client* client = ClientLibrary::LocalClientOrDie();
  for (char* arg : args) {
    HloSnapshot module;
    TF_CHECK_OK(
        tensorflow::ReadBinaryProto(tensorflow::Env::Default(), arg, &module));
    auto computation = client->LoadSnapshot(module).ConsumeValueOrDie();
    std::unique_ptr<ProgramShape> shape =
        client->GetComputationShape(computation).ConsumeValueOrDie();
    fprintf(stdout, "%s: %s :: %s\n", arg,
            module.hlo().hlo_module().name().c_str(),
            ShapeUtil::HumanString(*shape).c_str());
  }
}

}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  absl::Span<char* const> args(argv, argc);
  args.remove_prefix(1);  // Pop off the binary name, argv[0]
  xla::tools::RealMain(args);
  return 0;
}
