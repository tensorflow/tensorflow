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
// some_binary_snapshot_proto is obtained by serializing the HloSnapshot from
// ServiceInterface::SnapshotComputation to disk.
//
// Computations that require arguments can be replayed using fake data by
// passing --use_fake_data on the command line.  If the real data is available
// in the proto and --use_fake_data is false, the real data is used.
//
// Input can be a binary HloSnapshot proto, a binary HloProto proto, or a
// textual HLO string.
//
// The output format is:
//
// file_path: computation_name :: type:literal_str
//
// Note: If you pass multiple modules, they will be compiled in parallel but run
// in series.

#include <stdio.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/testing.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
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
  string fake_outfeed_shape;

  // generate_fake_infeed == true is a safe default: If the model has 0 or 1
  // infeeds, then it will work like normal.  If the model has more than one
  // infeed, it will be an error, but that wouldn't have worked anyway if you
  // hadn't passed generate_fake_infeed.
  //
  // Same for generate_fake_outfeed.
  bool generate_fake_infeed = true;
  bool generate_fake_outfeed = true;

  bool use_fake_data = false;
  bool print_result = true;
  int num_runs = 1;
};

StatusOr<std::unique_ptr<LocalExecutable>> CompileExecutable(
    const HloSnapshot& module, LocalClient* client) {
  XlaComputation computation(module.hlo().hlo_module());
  std::vector<Shape> argument_layouts;
  argument_layouts.reserve(
      computation.proto().host_program_shape().parameters_size());
  std::vector<const Shape*> argument_layout_ptrs;
  for (const ShapeProto& param :
       computation.proto().host_program_shape().parameters()) {
    argument_layouts.push_back(Shape(param));
    argument_layout_ptrs.push_back(&argument_layouts.back());
  }
  ExecutableBuildOptions exec_build_options;
  *exec_build_options.mutable_debug_options() = GetDebugOptionsFromFlags();
  return client->Compile(computation, argument_layout_ptrs, exec_build_options);
}

absl::optional<Shape> GetXfeedShape(bool is_infeed,
                                    const HloModuleProto& module,
                                    const Options& opts) {
  std::vector<HloInstructionProto> xfeed_instrs;
  for (const auto& comp : module.computations()) {
    for (const auto& instruction : comp.instructions()) {
      if (instruction.opcode() == HloOpcodeString(is_infeed
                                                      ? HloOpcode::kInfeed
                                                      : HloOpcode::kOutfeed)) {
        xfeed_instrs.push_back(instruction);
      }
    }
  }

  auto log_xfeed_instrs = [&] {
    for (const auto& infeed : xfeed_instrs) {
      LOG(ERROR) << "  " << ShapeUtil::HumanString(Shape(infeed.shape())) << " "
                 << infeed.name();
    }
  };

  auto find_instruction_from_id_or_die = [&](int64 id) {
    for (const auto& comp : module.computations()) {
      for (const auto& instruction : comp.instructions()) {
        if (instruction.id() == id) {
          return instruction;
        }
      }
    }
    LOG(FATAL) << "No instruction with id " << id;
  };

  absl::optional<Shape> xfeed_shape;
  string xfeed_name = is_infeed ? "infeed" : "outfeed";
  string fake_xfeed_shape =
      is_infeed ? opts.fake_infeed_shape : opts.fake_outfeed_shape;
  bool generate_fake_xfeed =
      is_infeed ? opts.generate_fake_infeed : opts.generate_fake_outfeed;
  if (!fake_xfeed_shape.empty()) {
    xfeed_shape = std::move(ParseShape(fake_xfeed_shape)).ValueOrDie();
  } else if (generate_fake_xfeed) {
    CHECK_LT(xfeed_instrs.size(), 2)
        << "--generate_fake_" << xfeed_name
        << " only works if the model has 0 or 1 " << xfeed_name << " ops.";
    if (xfeed_instrs.empty()) {
      LOG(INFO) << "Not generating fake " << xfeed_name
                << " shape; model has no " << xfeed_name << "s.";
    } else if (xfeed_instrs.size() == 1) {
      // kInfeed instructions should have a shape (buffer, token).  kOutfeed
      // instructions should have operand 0 of shape `buffer`. We want to xfeed
      // just `buffer`.
      xfeed_shape = is_infeed
                        ? Shape(xfeed_instrs.front().shape()).tuple_shapes(0)
                        : Shape(find_instruction_from_id_or_die(
                                    xfeed_instrs.front().operand_ids(0))
                                    .shape());
      LOG(INFO) << "Generating fake " << xfeed_name << " with inferred shape: "
                << ShapeUtil::HumanString(*xfeed_shape);
    } else {
      LOG(ERROR) << "--generate_fake_" << xfeed_name
                 << " only works if the model has 0 or 1 " << xfeed_name
                 << " ops, but this model has " << xfeed_instrs.size()
                 << " of them:";
      log_xfeed_instrs();
      LOG(FATAL) << "Can't run model with --generate_fake_infeed.";
    }
  } else if (!xfeed_instrs.empty()) {
    LOG(ERROR) << "Model contains " << xfeed_instrs.size() << " " << xfeed_name
               << " instruction(s), but neither --generate_fake_" << xfeed_name
               << " nor --fake_" << xfeed_name
               << "_shape was specified.  Execution will likely hang.";
    log_xfeed_instrs();
  }

  return xfeed_shape;
}

// Invokes the given computation passing arbitrary data for every (unbound)
// parameter if use_fake_data, Otherwise use recorded data if available.
//
// Similarly, infeeds fake data of shape fake_infeed_shape if it is provided.
// If generate_fake_infeed is true, the required infeed shape is derived from
// the computation and then used to provide a fake infeed shape.
//
// If neither generate_fake_infeed is true nor a fake_infeed_shape is provided,
// no infeed is performed.
StatusOr<Literal> ReplayComputation(const HloSnapshot& module,
                                    LocalExecutable* executable,
                                    LocalClient* client, const Options& opts) {
  XlaComputation computation(module.hlo().hlo_module());

  // Build the `argument_ptrs` vector, which contains ShapedBuffer*s to our
  // arguments.  This is a bit involved, because we may have to convert from
  // GlobalData to ShapedBuffer*, and we have to manage the lifetime of all our
  // objects.
  std::vector<ScopedShapedBuffer> scoped_shaped_buffer_arguments;
  std::vector<std::unique_ptr<GlobalData>> global_data_arguments;
  std::vector<const ShapedBuffer*> argument_ptrs;
  if (opts.use_fake_data) {
    // Run fake computations with debug options ignoring XLA_FLAGS.  Users very
    // likely want XLA_FLAGS only to apply to the "real" computation being run,
    // not to the fake computations we use for generating arguments.
    auto debug_opts = DefaultDebugOptionsIgnoringFlags();
    global_data_arguments =
        MakeFakeArgumentsOrDie(computation, client, &debug_opts);
    for (const auto& data : global_data_arguments) {
      argument_ptrs.push_back(
          client->GlobalDataToShapedBuffer(data->handle(), /*device_ordinal=*/0)
              .ValueOrDie());
    }
  } else {  // use recorded data if available
    for (const auto& proto : module.arguments()) {
      TF_ASSIGN_OR_RETURN(Literal literal, Literal::CreateFromProto(proto));
      TF_ASSIGN_OR_RETURN(
          ScopedShapedBuffer data,
          client->LiteralToShapedBuffer(literal, /*device_ordinal=*/0));
      scoped_shaped_buffer_arguments.push_back(std::move(data));
    }
    for (const auto& argument : scoped_shaped_buffer_arguments) {
      argument_ptrs.push_back(&argument);
    }
  }

  if (absl::optional<Shape> infeed_shape = GetXfeedShape(
          /*is_infeed=*/true, computation.proto(), opts)) {
    auto infeed_data = std::make_shared<Literal>(
        std::move(MakeFakeLiteral(*infeed_shape)).ValueOrDie());
    xla::gpu::GetOrCreateInfeedManager()
        ->RegisterBeforeGetNextDestinationCallback([infeed_data, client] {
          TF_CHECK_OK(client->TransferToInfeed(*infeed_data));
        });
  }

  absl::optional<tensorflow::thread::ThreadPool> outfeed_thread_pool;
  if (absl::optional<Shape> outfeed_shape = GetXfeedShape(
          /*is_infeed=*/false, computation.proto(), opts)) {
    // For each an outfeed that runs, enqueue a task that will consume it.  We
    // need a thread pool because the act of running an outfeed blocks on there
    // being a destination available, and the act of making a destination
    // available blocks on there being outfeed data available.
    outfeed_thread_pool.emplace(tensorflow::Env::Default(), "infeed",
                                /*num_threads=*/1);
    auto consume_outfeed = [client, outfeed_shape] {
      TF_CHECK_OK(
          client->TransferFromOutfeedLocal(*outfeed_shape, /*device_ordinal=*/0)
              .status());
      VLOG(1) << "Received outfeed data of shape "
              << ShapeUtil::HumanStringWithLayout(*outfeed_shape);
    };
    xla::gpu::GetOrCreateOutfeedManager()
        ->RegisterBeforeGetNextDestinationCallback(
            [consume_outfeed, &outfeed_thread_pool] {
              outfeed_thread_pool->Schedule(consume_outfeed);
            });
  }

  // Do not attempt to run the executable if num_runs is less than 1.
  if (opts.num_runs < 1) {
    return Cancelled("Cancelled after compilation since --num_runs < 1.");
  }

  // Run the computation num_runs times, and return the result from the last
  // execution.
  const bool xla_hlo_profile = GetDebugOptionsFromFlags().xla_hlo_profile();
  StreamExecutorMemoryAllocator allocator(
      client->platform(),
      {client->platform()->ExecutorForDevice(0).ValueOrDie()});
  absl::optional<ScopedShapedBuffer> final_result;
  for (int i = 0; i < opts.num_runs; ++i) {
    // If xla_hlo_profile is enabled, print a noisy message before the last run,
    // making it easier to separate this profile from the others in the logspam.
    bool is_final_result = i == opts.num_runs - 1;
    if (xla_hlo_profile && is_final_result) {
      LOG(INFO) << "\n\n***** Final run below ******";
    }
    ExecutionProfile profile;
    ExecutableRunOptions run_options;
    run_options.set_execution_profile(&profile);
    run_options.set_allocator(&allocator);

    TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result,
                        executable->Run(argument_ptrs, run_options));
    LOG(INFO) << "Done executing in "
              << static_cast<double>(profile.compute_time_ns()) / 1e9
              << "s: " << module.hlo().hlo_module().name();

    // Save the result if this is for the final iteration.  Otherwise discard
    // the result before rerunning the computation, so as to free up the
    // relevant memory.
    if (is_final_result) {
      final_result = std::move(result);
    }
  }

  TF_ASSIGN_OR_RETURN(Literal result_literal,
                      client->ShapedBufferToLiteral(*final_result));
  return result_literal;
}

StatusOr<HloSnapshot> ParseInputFile(const string& filename,
                                     const Options& opts) {
  tensorflow::Env* env = tensorflow::Env::Default();
  HloSnapshot snapshot;
  auto s = tensorflow::ReadBinaryProto(env, filename, &snapshot);
  if (s.ok()) {
    return snapshot;
  }
  if (s.code() == tensorflow::error::NOT_FOUND) {
    return s;
  }
  CHECK(opts.use_fake_data)
      << "Without --use_fake_data, you must pass an HloSnapshot -- HloProto "
         "and textual HLO don't carry real data.";
  fprintf(stderr, "%s: is not HloSnapshot. Trying HloProto.\n",
          filename.c_str());

  if (tensorflow::ReadBinaryProto(env, filename, snapshot.mutable_hlo()).ok()) {
    return snapshot;
  }
  fprintf(stderr, "%s: is not HloProto. Trying HLO text.\n", filename.c_str());
  string contents;
  TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(env, filename, &contents));
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsFromFlags());
  StatusOr<std::unique_ptr<HloModule>> module =
      ParseHloString(contents, config);
  if (module.ok()) {
    *snapshot.mutable_hlo()->mutable_hlo_module() =
        module.ValueOrDie()->ToProto();
    return snapshot;
  }
  fprintf(stderr, "%s: is not HLO text.  Nothing left to try.\n",
          filename.c_str());
  return InvalidArgument("Could not parse %s.", filename);
}

int RealMain(absl::Span<char* const> args, const Options& opts) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();
  int exit_status = EXIT_SUCCESS;

  std::vector<HloSnapshot> snapshots;
  for (char* arg : args) {
    StatusOr<HloSnapshot> maybe_snapshot = ParseInputFile(arg, opts);
    if (maybe_snapshot.ok()) {
      snapshots.push_back(std::move(maybe_snapshot).ValueOrDie());
    } else {
      LOG(ERROR) << "Can't handle file " << arg << ": "
                 << maybe_snapshot.status();
    }
  }

  // Compile all the modules in parallel.
  LOG(INFO) << "Compiling " << snapshots.size() << " modules in parallel.";
  std::vector<StatusOr<std::unique_ptr<LocalExecutable>>> executables;
  {
    // ThreadPool CHECK-fails if we give it 0 threads.
    tensorflow::thread::ThreadPool thread_pool(
        tensorflow::Env::Default(), tensorflow::ThreadOptions(),
        "compile_modules", std::max(size_t{1}, snapshots.size()),
        /*low_latency_hint=*/false);
    executables.resize(snapshots.size());
    for (int64 i = 0; i < snapshots.size(); ++i) {
      thread_pool.Schedule([&snapshots, &executables, client, i] {
        executables[i] = CompileExecutable(snapshots[i], client);
      });
    }
  }
  LOG(INFO) << "Done compiling; now running the modules.";

  for (int64 i = 0; i < executables.size(); ++i) {
    if (!executables[i].ok()) {
      LOG(ERROR) << "Compilation failed: " << executables[i].status();
      exit_status = EXIT_FAILURE;
      continue;
    }
    LocalExecutable* executable = executables[i].ValueOrDie().get();
    LOG(ERROR) << "Running iteration " << i;
    StatusOr<Literal> result_status =
        ReplayComputation(snapshots[i], executable, client, opts);
    LOG(ERROR) << "iteration complete.";
    if (!result_status.ok()) {
      fprintf(stderr, "%s: error: %s\n", args[i],
              result_status.status().ToString().c_str());
      exit_status = EXIT_FAILURE;
      continue;
    }

    if (opts.print_result) {
      Literal result = std::move(result_status).ValueOrDie();
      fprintf(stdout, "%s: %s :: %s:%s\n", args[i],
              executable->executable()->module().name().c_str(),
              ShapeUtil::HumanString(result.shape()).c_str(),
              result.ToString().c_str());
      auto& snapshot = snapshots[i];
      if (snapshot.has_result()) {
        Literal literal =
            Literal::CreateFromProto(snapshot.result()).ConsumeValueOrDie();
        fprintf(
            stdout, "was %s:%s\n",
            ShapeUtil::HumanString(Shape(snapshot.result().shape())).c_str(),
            literal.ToString().c_str());
      }
    }
  }

  ClientLibrary::DestroyLocalInstances();
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
      tensorflow::Flag("fake_outfeed_shape", &opts.fake_outfeed_shape,
                       "Shape of fake data to outfeed from computation"),
      tensorflow::Flag("generate_fake_infeed", &opts.generate_fake_infeed,
                       "Whether a fake infeed shape should be derived "
                       "from the computation"),
      tensorflow::Flag("generate_fake_outfeed", &opts.generate_fake_outfeed,
                       "Whether a fake outfeed shape should be derived "
                       "from the computation"),
  };
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc < 2 || !parse_ok) {
    LOG(QFATAL) << usage;
  }

  absl::Span<char* const> args(argv, argc);
  args.remove_prefix(1);  // Pop off the binary name, argv[0]
  return xla::tools::RealMain(args, opts);
}
