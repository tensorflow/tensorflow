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

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {
namespace se = ::perftools::gputools;

class HloProfileTest : public ClientLibraryTestBase {};

struct ParsedProfileOutputLine {
  int64 cycles;
  string cycles_percentage;
  double usec;
  string flops;
  string trops;
  string bytes_per_sec;
  string bytes_per_cycle;
  string name;
};

StatusOr<ParsedProfileOutputLine> ParseProfileOutputLine(const string& line,
                                                         bool expect_flops,
                                                         bool expect_trops) {
  string separator = "[^:]*:: +";
  string match_percentage = "\\d+\\.\\d\\d%";
  string match_cycles = "(\\d+) cycles +\\( *(" + match_percentage + ")\\)";
  string match_usecs = "([0-9.]+) usec";
  string match_flops = expect_flops ? "([0-9.TGMk]+)FLOP/s" : "(<none>)";
  string match_trops = expect_trops ? "([0-9.TGMk]+)TROP/s" : "(<none>)";
  string match_bytes_per_sec = "([0-9.TGMKi]+)B/s";
  string match_bytes_per_cycle = "([0-9.TGMKi]+)B/cycle";
  string regexp_pattern = tensorflow::strings::StrCat(
      " +", match_cycles, separator, match_usecs, separator, match_flops,
      separator, match_trops, separator, match_bytes_per_sec, separator,
      match_bytes_per_cycle, separator, "(.*)");

  RE2 pattern(regexp_pattern);
  ParsedProfileOutputLine parsed_line;
  bool matched = RE2::FullMatch(
      line, pattern, &parsed_line.cycles, &parsed_line.cycles_percentage,
      &parsed_line.usec, &parsed_line.flops, &parsed_line.trops,
      &parsed_line.bytes_per_sec, &parsed_line.bytes_per_cycle,
      &parsed_line.name);
  if (!matched) {
    return tensorflow::errors::InvalidArgument(
        "Input did not match regexp.  Input: ", line,
        ", Regexp: ", regexp_pattern);
  }

  return parsed_line;
}

// Returns void so that we can ASSERT.
void ExecuteAndFetchProfile(string* profile_output, LocalClient* client,
                            const Computation& computation,
                            const Shape& lhs_arg_shape,
                            const Shape& rhs_arg_shape) {
  LocalService* service = ClientLibrary::GetXlaService(client->platform());
  Backend* backend = service->mutable_backend();
  se::StreamExecutor* executor = backend->default_stream_executor();
  DeviceMemoryAllocator* allocator = backend->memory_allocator();
  auto* transfer_manager = backend->transfer_manager();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ScopedShapedBuffer> lhs_arg,
      transfer_manager->AllocateScopedShapedBuffer(
          lhs_arg_shape, allocator, backend->default_device_ordinal()));
  TF_ASSERT_OK(transfer_manager->TransferLiteralToDevice(
      executor, *Literal::CreateFromShape(lhs_arg_shape), *lhs_arg));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ScopedShapedBuffer> rhs_arg,
      transfer_manager->AllocateScopedShapedBuffer(
          rhs_arg_shape, allocator, backend->default_device_ordinal()));
  TF_ASSERT_OK(transfer_manager->TransferLiteralToDevice(
      executor, *Literal::CreateFromShape(rhs_arg_shape), *rhs_arg));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LocalExecutable> local_executable,
      client->Compile(computation, {&lhs_arg_shape, &rhs_arg_shape},
                      ExecutableBuildOptions()));

  Executable* executable = local_executable->executable();
  HloExecutionProfile hlo_execution_profile(
      &executable->hlo_profile_printer(), &executable->hlo_profile_index_map());

  TF_ASSERT_OK_AND_ASSIGN(
      Backend::StreamPtr stream_ptr,
      backend->BorrowStream(backend->default_device_ordinal()));
  ExecutableRunOptions exec_run_options;
  exec_run_options.set_stream(stream_ptr.get());
  exec_run_options.set_allocator(backend->memory_allocator());
  exec_run_options.set_intra_op_thread_pool(
      backend->eigen_intra_op_thread_pool_device());
  ServiceExecutableRunOptions run_options(
      exec_run_options, /*borrow_stream=*/nullptr,
      backend->eigen_intra_op_thread_pool());
  TF_ASSERT_OK_AND_ASSIGN(
      auto execution_result,
      executable->ExecuteOnStream(&run_options, {lhs_arg.get(), rhs_arg.get()},
                                  &hlo_execution_profile));
  (void)execution_result;

  *profile_output =
      hlo_execution_profile.ToString(executor->GetDeviceDescription());

  XLA_VLOG_LINES(4, *profile_output);
}

// TODO(b/71364943): This test exposes a bug in the parallel CPU backend.
XLA_TEST_F(HloProfileTest, DISABLED_ON_CPU_PARALLEL(ProfileSingleComputation)) {
  const int64 m = 256, k = 256, n = 256;
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {m, k});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {m, k});

  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetDefaultPlatform());
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client,
                          ClientLibrary::GetOrCreateLocalClient(platform));

  ComputationBuilder builder(client, TestName());
  auto result = builder.Tanh(builder.Dot(
      builder.Parameter(0, ShapeUtil::MakeShape(F32, {m, k}), "dot_lhs"),
      builder.Parameter(1, ShapeUtil::MakeShape(F32, {k, n}), "dot_rhs")));

  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());

  string profile_output;
  ExecuteAndFetchProfile(&profile_output, client, computation, lhs_shape,
                         rhs_shape);

  std::vector<string> profile_output_lines =
      tensorflow::str_util::Split(profile_output, '\n');

  TF_ASSERT_OK_AND_ASSIGN(
      ParsedProfileOutputLine total_profile,
      ParseProfileOutputLine(profile_output_lines[1], /*expect_flops=*/true,
                             /*expect_trops=*/true));

  TF_ASSERT_OK_AND_ASSIGN(
      ParsedProfileOutputLine dot_profile,
      ParseProfileOutputLine(profile_output_lines[2], /*expect_flops=*/true,
                             /*expect_trops=*/false));

  TF_ASSERT_OK_AND_ASSIGN(
      ParsedProfileOutputLine tanh_profile,
      ParseProfileOutputLine(profile_output_lines[3], /*expect_flops=*/false,
                             /*expect_trops=*/true));

  EXPECT_GT(total_profile.cycles, 0);
  EXPECT_EQ(total_profile.cycles_percentage, "100.00%");

  EXPECT_GT(total_profile.cycles, dot_profile.cycles);
  EXPECT_NE(dot_profile.cycles_percentage, "0.00%");
  EXPECT_NE(dot_profile.cycles_percentage, "100.00%");

  EXPECT_GT(total_profile.cycles, tanh_profile.cycles);
  EXPECT_NE(tanh_profile.cycles_percentage, "0.00%");
  EXPECT_NE(tanh_profile.cycles_percentage, "100.00%");
}

// TODO(b/71364943): This test exposes a bug in the parallel CPU backend.
//
// TODO(b/71544591): The GPU backend does not record cycles spent in on Hlo
// instructions "interior" to while nodes.
XLA_TEST_F(HloProfileTest,
           DISABLED_ON_GPU(DISABLED_ON_CPU_PARALLEL(ProfileWhileComputation))) {
  const int64 size = 256;
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {size, size});
  Shape while_result_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {}), matrix_shape});

  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetDefaultPlatform());
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client,
                          ClientLibrary::GetOrCreateLocalClient(platform));

  Computation condition;
  {
    ComputationBuilder builder(client, "condition");
    auto state = builder.Parameter(0, while_result_shape, "state");
    auto iteration = builder.GetTupleElement(state, 0);
    builder.Gt(builder.ConstantR0<int32>(5), iteration);
    TF_ASSERT_OK_AND_ASSIGN(condition, builder.Build());
  }

  Computation body;
  {
    ComputationBuilder builder(client, "body");
    auto state = builder.Parameter(0, while_result_shape, "state");
    auto matrix = builder.GetTupleElement(state, 1);
    auto next_iteration = builder.Add(builder.GetTupleElement(state, 0),
                                      builder.ConstantR0<int32>(1));
    builder.Tuple({next_iteration, builder.Dot(matrix, matrix)});
    TF_ASSERT_OK_AND_ASSIGN(body, builder.Build());
  }

  ComputationBuilder builder(client, TestName());
  auto initial_while_state =
      builder.Tuple({builder.ConstantR0<int32>(0),
                     builder.Parameter(0, matrix_shape, "initial_value")});
  auto while_result = builder.While(condition, body, initial_while_state);
  builder.Add(builder.GetTupleElement(while_result, 1),
              builder.Parameter(1, matrix_shape, "other_value"));

  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());

  string profile_output;
  ExecuteAndFetchProfile(&profile_output, client, computation, matrix_shape,
                         matrix_shape);

  std::vector<string> profile_output_lines =
      tensorflow::str_util::Split(profile_output, '\n');

  auto while_body_profile_start =
      std::find_if(profile_output_lines.begin(), profile_output_lines.end(),
                   [](tensorflow::StringPiece s) {
                     return s.starts_with("Execution profile for body");
                   });

  ASSERT_NE(while_body_profile_start, profile_output_lines.end());

  TF_ASSERT_OK_AND_ASSIGN(
      ParsedProfileOutputLine total_while_body_profile,
      ParseProfileOutputLine(*std::next(while_body_profile_start, 1),
                             /*expect_flops=*/false,
                             /*expect_trops=*/false));

  TF_ASSERT_OK_AND_ASSIGN(
      ParsedProfileOutputLine dot_profile,
      ParseProfileOutputLine(*std::next(while_body_profile_start, 2),
                             /*expect_flops=*/false,
                             /*expect_trops=*/false));

  EXPECT_GT(total_while_body_profile.cycles, 0);
  EXPECT_EQ(total_while_body_profile.name, "[total]");
  EXPECT_EQ(total_while_body_profile.cycles_percentage, "100.00%");

  EXPECT_GT(total_while_body_profile.cycles, dot_profile.cycles);
  EXPECT_NE(dot_profile.cycles_percentage, "0.00%");
  EXPECT_NE(dot_profile.cycles_percentage, "100.00%");
}
}  // namespace
}  // namespace xla

static std::pair<int, char**> AddXlaHloProfileFlag(int argc, char** argv) {
  // Intentional "leak".
  char** new_argv = new char*[argc + 2];
  for (int i = 0; i < argc; i++) {
    new_argv[i] = argv[i];
  }

  // We do it this way (as opposed to piping in a modified DebugOptions
  // instance) for better end-to-end integration testing.
  new_argv[argc] = strdup("--xla_hlo_profile");

  // Fusion can change the Hlo instructions that show up in the final Hlo
  // executable, so block it here.
  new_argv[argc + 1] = strdup("--xla_disable_hlo_passes=fusion");
  return {argc + 2, new_argv};
}

GTEST_API_ int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendDebugOptionsFlags(&flag_list);
  std::tie(argc, argv) = AddXlaHloProfileFlag(argc, argv);

  auto usage = tensorflow::Flags::Usage(argv[0], flag_list);
  if (!tensorflow::Flags::Parse(&argc, argv, flag_list)) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }

  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
