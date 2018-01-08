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
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace tools {
namespace {

// Command-line opts to this tool.  See main() for descriptions of these
// fields.
struct Options {
  string fake_infeed_shape;
  bool use_fake_data = false;
  bool print_result = true;
  int num_runs = 1;
};

// Invokes the given computation passing arbitrary data for every (unbound)
// parameter if use_fake_data, Otherwise use recorded data if available.
//
// Similarly, infeeds fake data of shape fake_infeed_shape if it is provided;
// otherwise, no infeed is performed.
StatusOr<std::unique_ptr<Literal>> ReplayComputation(
    const SessionModule& module, Client* client, const Options& opts) {
  TF_ASSIGN_OR_RETURN(Computation computation, client->LoadSnapshot(module));

  std::vector<std::unique_ptr<GlobalData>> arguments;
  if (opts.use_fake_data) {
    arguments = MakeFakeArgumentsOrDie(computation, client);
  } else {  // use recorded data if available
    for (const auto& proto : module.arguments()) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::Literal> literal,
                          Literal::CreateFromProto(proto));
      TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> data,
                          client->TransferToServer(*literal));
      arguments.push_back(std::move(data));
    }
  }

  // We only instantiate the thread pool if the user has requested that a
  // concurrent infeed occur via the fake_infeed_shape.
  tensorflow::gtl::optional<tensorflow::thread::ThreadPool> pool;

  if (!opts.fake_infeed_shape.empty()) {
    pool.emplace(tensorflow::Env::Default(), "infeed",
                 /*num_threads=*/1);
    pool->Schedule([opts, client]() {
      StatusOr<Shape> shape_status =
          ShapeUtil::ParseShapeString(opts.fake_infeed_shape);
      TF_CHECK_OK(shape_status.status());
      Shape shape = std::move(shape_status).ValueOrDie();
      StatusOr<std::unique_ptr<Literal>> data_status = MakeFakeLiteral(shape);
      TF_CHECK_OK(data_status.status());
      std::unique_ptr<Literal> data = std::move(data_status).ValueOrDie();
      while (true) {
        TF_CHECK_OK(client->TransferToInfeed(*data));
      }
    });
  }

  std::vector<GlobalData*> execute_arguments;
  execute_arguments.reserve(arguments.size());
  for (auto& argument : arguments) {
    execute_arguments.push_back(argument.get());
  }

  // Run the computation num_runs times, and return the result from the last
  // execution.
  std::unique_ptr<Literal> result;
  for (int i = 0; i < opts.num_runs; ++i) {
    ExecutionProfile profile;
    if (opts.print_result) {
      TF_ASSIGN_OR_RETURN(result, client->ExecuteAndTransfer(
                                      computation, execute_arguments,
                                      /*execution_options=*/nullptr, &profile));
    } else {
      // If we're not printing the result, execute the computation but don't
      // bother retrieving the result.  This can be a significant speedup.
      TF_RETURN_IF_ERROR(client
                             ->Execute(computation, execute_arguments,
                                       /*execution_options=*/nullptr, &profile)
                             .status());
    }
    LOG(INFO) << "Execution took "
              << static_cast<double>(profile.compute_time_ns()) / 1e9 << "s";
  }

  return std::move(result);
}

int RealMain(tensorflow::gtl::ArraySlice<char*> args, const Options& opts) {
  Client* client = ClientLibrary::LocalClientOrDie();
  tensorflow::Env* env = tensorflow::Env::Default();
  int exit_status = EXIT_SUCCESS;
  for (char* arg : args) {
    SessionModule module;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(env, arg, &module));
    StatusOr<std::unique_ptr<Literal>> result_status =
        ReplayComputation(module, client, opts);
    if (!result_status.ok()) {
      fprintf(stderr, "%s: error: %s\n", arg,
              result_status.status().ToString().c_str());
      exit_status = EXIT_FAILURE;
      continue;
    }

    std::unique_ptr<Literal> result = result_status.ConsumeValueOrDie();
    if (result != nullptr) {
      fprintf(stdout, "%s: %s :: %s:%s\n", arg, module.entry().name().c_str(),
              ShapeUtil::HumanString(result->shape()).c_str(),
              result->ToString().c_str());
      if (module.has_result()) {
        std::unique_ptr<Literal> literal =
            Literal::CreateFromProto(module.result()).ConsumeValueOrDie();
        fprintf(stdout, "was %s:%s\n",
                ShapeUtil::HumanString(module.result().shape()).c_str(),
                literal->ToString().c_str());
      }
    }
  }
  return exit_status;
}

}  // namespace
}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  xla::tools::Options opts;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("use_fake_data", &opts.use_fake_data,
                       "Replay computation using fake data"),
      tensorflow::Flag("print_result", &opts.print_result,
                       "Print the result of the computation to stdout"),
      tensorflow::Flag("num_runs", &opts.num_runs,
                       "Number of times to run each computation"),
      tensorflow::Flag("fake_infeed_shape", &opts.fake_infeed_shape,
                       "Shape of fake data to construct for (infinite) infeed"),
  };
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc < 2 || !parse_ok) {
    LOG(QFATAL) << usage;
  }

  tensorflow::gtl::ArraySlice<char*> args(argv, argc);
  args.pop_front();  // Pop off the binary name, argv[0]
  return xla::tools::RealMain(args, opts);
}
