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

// Usage: dumped_computation_to_tf_graph some_binary_snapshot_proto*
//
// Dumps a tensorflow GraphDef in text format for a snapshot computation. The
// dumped graph is an HLO computation with HLO instructions as nodes and can be
// visualized on Tensorboard. Upload the dumped files on Tensorboard.
//
// some_binary_snapshot_proto is obtained by serializing the SessionModule from
// ServiceInterface::SnapshotComputation to disk.

#include <stdio.h>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

using tensorflow::Env;

namespace xla {
namespace tools {

void RealMain(tensorflow::gtl::ArraySlice<char*> args) {
  Client* client = ClientLibrary::LocalClientOrDie();
  for (char* arg : args) {
    HloSnapshot module;
    TF_CHECK_OK(
        tensorflow::ReadBinaryProto(tensorflow::Env::Default(), arg, &module));
    XlaComputation computation =
        client->LoadSnapshot(module).ConsumeValueOrDie();
    DebugOptions debug_options = legacy_flags::GetDebugOptionsFromFlags();
    debug_options.set_xla_generate_hlo_graph(".*");
    debug_options.set_xla_hlo_dump_as_graphdef(true);
    ComputationStats stats =
        client->GetComputationStats(computation, debug_options)
            .ConsumeValueOrDie();
    fprintf(stdout, ">>> %s :: %s\n", arg, stats.DebugString().c_str());
  }
}

}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendDebugOptionsFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::gtl::ArraySlice<char*> args(argv, argc);
  args.pop_front();  // Pop off the binary name, argv[0]
  xla::tools::RealMain(args);
  return 0;
}
