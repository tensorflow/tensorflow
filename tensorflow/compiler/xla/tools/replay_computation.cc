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

// Usage: replay_computation some_binary_snapshot_proto*
//
// Replays computations and shows the results on the command line.
//
// some_binary_snapshot_proto is obtained by serializing the SessionModule from
// ServiceInterface::SnapshotComputation to disk.
//
// Computations that require arguments can be replayed using fake data by
// passing --use_fake_data on the command line.  If the real data is available
// in the proto and --use_fake_data is false, the real data is used.
//
// The output format is:
//
// file_path: computation_name :: type:literal_str

#include <stdio.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/testing.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace tools {

// Invokes the given computation passing arbitrary data for every (unbound)
// parameter if use_fake_data, Otherwise use recorded data if available.
StatusOr<std::unique_ptr<Literal>> ReplayComputation(
    const SessionModule& module, bool use_fake_data, Client* client) {
  TF_ASSIGN_OR_RETURN(Computation computation, client->LoadSnapshot(module));

  std::vector<std::unique_ptr<GlobalData>> arguments;
  if (use_fake_data) {
    arguments = MakeFakeArgumentsOrDie(computation, client);
  } else {  // use recorded data if available
    for (const auto& proto : module.arguments()) {
      Literal literal(proto);
      TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> data,
                          client->TransferToServer(literal));
      arguments.push_back(std::move(data));
    }
  }

  std::vector<GlobalData*> execute_arguments;
  execute_arguments.reserve(arguments.size());
  for (auto& argument : arguments) {
    execute_arguments.push_back(argument.get());
  }
  return client->ExecuteAndTransfer(computation, execute_arguments);
}

void RealMain(tensorflow::gtl::ArraySlice<char*> args, bool use_fake_data) {
  Client* client = ClientLibrary::LocalClientOrDie();
  tensorflow::Env* env = tensorflow::Env::Default();
  for (char* arg : args) {
    SessionModule module;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(env, arg, &module));
    StatusOr<std::unique_ptr<Literal>> result_status =
        ReplayComputation(module, use_fake_data, client);
    if (!result_status.ok()) {
      fprintf(stderr, "%s: error: %s\n", arg,
              result_status.status().ToString().c_str());
      continue;
    }
    std::unique_ptr<Literal> result = result_status.ConsumeValueOrDie();
    fprintf(stdout, "%s: %s :: %s:%s\n", arg, module.entry().name().c_str(),
            ShapeUtil::HumanString(result->shape()).c_str(),
            LiteralUtil::ToString(*result).c_str());
    if (module.has_result()) {
      fprintf(stdout, "was %s:%s\n",
              ShapeUtil::HumanString(module.result().shape()).c_str(),
              LiteralUtil::ToString(Literal(module.result())).c_str());
    }
  }
}

}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  // Flags
  bool use_fake_data = false;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("use_fake_data", &use_fake_data,
                       "Replay computation using fake data"),
  };
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc < 2 || !parse_ok) {
    LOG(QFATAL) << usage;
  }

  tensorflow::gtl::ArraySlice<char*> args(argv, argc);
  args.pop_front();  // Pop off the binary name, argv[0]
  xla::tools::RealMain(args, use_fake_data);
  return 0;
}
