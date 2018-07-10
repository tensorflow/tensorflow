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
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

namespace gtl = ::tensorflow::gtl;

class HloProfileTest : public ClientLibraryTestBase {};

struct ParsedProfileOutputLine {
  int64 cycles;
  string cycles_percentage;
  double usec;
  string flops;
  string trops;
  string bytes_per_sec;
  string bytes_per_cycle;
  string opcode;
};

::testing::AssertionResult HasFlops(
    const ParsedProfileOutputLine& parsed_line) {
  if (RE2::FullMatch(parsed_line.flops, "[0-9.TGMk]+FLOP/s")) {
    return ::testing::AssertionSuccess()
           << "'flops' field present in  " << parsed_line.opcode << ": '"
           << parsed_line.flops << "'";
  }

  return ::testing::AssertionFailure()
         << "'flops' field absent in  " << parsed_line.opcode << ": '"
         << parsed_line.flops << "'";
}

::testing::AssertionResult HasTrops(
    const ParsedProfileOutputLine& parsed_line) {
  if (RE2::FullMatch(parsed_line.trops, "[0-9.TGMk]+TROP/s")) {
    return ::testing::AssertionSuccess()
           << "'trops' field present in  " << parsed_line.opcode << ": '"
           << parsed_line.trops << "'";
  }

  return ::testing::AssertionFailure()
         << "'trops' field absent in  " << parsed_line.opcode << ": '"
         << parsed_line.trops << "'";
}

Status ParseOneProfileOutputLine(
    const string& line, bool expect_hlo,
    gtl::FlatMap<string, ParsedProfileOutputLine>* parsed_results,
    tensorflow::gtl::ArraySlice<tensorflow::StringPiece> opcodes_to_ignore =
        {}) {
  string separator = "[^:]*:: +";
  string match_percentage = "\\d+\\.\\d\\d%";
  string match_cycles = "(\\d+) cycles +\\( *(" + match_percentage + ")\\)";
  string match_usecs = "([0-9.]+) usec";
  string match_flops = "([^ ]*)";
  string match_trops = "([^ ]*)";
  string match_bytes_per_sec = "([0-9.TGMKi]+)B/s";
  string match_bytes_per_cycle = "([0-9.TGMKi]+)B/cycle";

  // The underlined part is what we're trying to match with match_opcode:
  //
  //   %dot33 = f32[256,256]{1,0} dot(...)
  //                              ^^^

  string match_opcode =
      expect_hlo ? "%[^=]+= [^ ]+ ([^(]+)\\(.*" : "(\\[total\\])";
  string regexp_pattern = tensorflow::strings::StrCat(
      " +", match_cycles, separator, match_usecs, separator, match_flops,
      separator, match_trops, separator, match_bytes_per_sec, separator,
      match_bytes_per_cycle, separator, match_opcode);

  ParsedProfileOutputLine parsed_line;
  bool matched = RE2::FullMatch(
      line, regexp_pattern, &parsed_line.cycles, &parsed_line.cycles_percentage,
      &parsed_line.usec, &parsed_line.flops, &parsed_line.trops,
      &parsed_line.bytes_per_sec, &parsed_line.bytes_per_cycle,
      &parsed_line.opcode);
  if (!matched) {
    return tensorflow::errors::InvalidArgument(
        "Input did not match regexp.  Input: ", line,
        ", Regexp: ", regexp_pattern);
  }

  if (!c_linear_search(opcodes_to_ignore, parsed_line.opcode)) {
    InsertOrDie(parsed_results, parsed_line.opcode, parsed_line);
  }

  return Status::OK();
}

// Returns void so that we can ASSERT.
void ExecuteAndFetchProfile(string* profile_output, LocalClient* client,
                            const XlaComputation& computation,
                            const Shape& lhs_arg_shape,
                            const Shape& rhs_arg_shape) {
  LocalService* service = ClientLibrary::GetXlaService(client->platform());
  Backend* backend = service->mutable_backend();
  se::StreamExecutor* executor = backend->default_stream_executor();
  DeviceMemoryAllocator* allocator = backend->memory_allocator();
  auto* transfer_manager = backend->transfer_manager();
  TF_ASSERT_OK_AND_ASSIGN(
      Backend::StreamPtr stream_ptr,
      backend->BorrowStream(backend->default_device_ordinal()));

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer lhs_arg,
      transfer_manager->AllocateScopedShapedBuffer(
          lhs_arg_shape, allocator, backend->default_device_ordinal()));
  TF_ASSERT_OK(transfer_manager->TransferLiteralToDevice(
      stream_ptr.get(), *Literal::CreateFromShape(lhs_arg_shape), lhs_arg));

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer rhs_arg,
      transfer_manager->AllocateScopedShapedBuffer(
          rhs_arg_shape, allocator, backend->default_device_ordinal()));
  TF_ASSERT_OK(transfer_manager->TransferLiteralToDevice(
      stream_ptr.get(), *Literal::CreateFromShape(rhs_arg_shape), rhs_arg));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LocalExecutable> local_executable,
      client->Compile(computation, {&lhs_arg_shape, &rhs_arg_shape},
                      ExecutableBuildOptions().set_hlo_profile(true)));

  Executable* executable = local_executable->executable();
  HloExecutionProfile hlo_execution_profile(
      &executable->hlo_profile_printer_data(),
      &executable->hlo_profile_index_map());

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
      executable->ExecuteOnStream(&run_options, {&lhs_arg, &rhs_arg},
                                  &hlo_execution_profile));
  (void)execution_result;

  *profile_output =
      hlo_execution_profile.ToString(executor->GetDeviceDescription());

  XLA_VLOG_LINES(4, *profile_output);
}

XLA_TEST_F(HloProfileTest, ProfileSingleComputation) {
  const int64 m = 256, k = 256, n = 256;
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {m, k});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {m, k});

  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetDefaultPlatform());
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client,
                          ClientLibrary::GetOrCreateLocalClient(platform));

  XlaBuilder builder(TestName());
  Tanh(Add(
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {m, k}), "dot_lhs"),
      Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {k, n}), "dot_rhs")));

  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());

  string profile_output;
  ExecuteAndFetchProfile(&profile_output, client, computation, lhs_shape,
                         rhs_shape);

  std::vector<string> profile_output_lines =
      tensorflow::str_util::Split(profile_output, '\n');

  gtl::FlatMap<string, ParsedProfileOutputLine> parsed_profile_lines;

  TF_ASSERT_OK(ParseOneProfileOutputLine(
      profile_output_lines[1], /*expect_hlo=*/false, &parsed_profile_lines));

  TF_ASSERT_OK(ParseOneProfileOutputLine(
      profile_output_lines[2], /*expect_hlo=*/true, &parsed_profile_lines));

  TF_ASSERT_OK(ParseOneProfileOutputLine(
      profile_output_lines[3], /*expect_hlo=*/true, &parsed_profile_lines));

  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine total_profile,
                          MaybeFind(parsed_profile_lines, "[total]"));
  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine dot_profile,
                          MaybeFind(parsed_profile_lines, "add"));
  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine tanh_profile,
                          MaybeFind(parsed_profile_lines, "tanh"));

  EXPECT_GT(total_profile.cycles, 0);
  EXPECT_EQ(total_profile.cycles_percentage, "100.00%");

  EXPECT_TRUE(HasFlops(total_profile));
  EXPECT_TRUE(HasTrops(total_profile));

  EXPECT_GT(total_profile.cycles, dot_profile.cycles);
  EXPECT_NE(dot_profile.cycles_percentage, "0.00%");
  EXPECT_NE(dot_profile.cycles_percentage, "100.00%");

  EXPECT_TRUE(HasFlops(dot_profile));
  EXPECT_FALSE(HasTrops(dot_profile));

  EXPECT_GT(total_profile.cycles, tanh_profile.cycles);
  EXPECT_NE(tanh_profile.cycles_percentage, "0.00%");
  EXPECT_NE(tanh_profile.cycles_percentage, "100.00%");

  EXPECT_FALSE(HasFlops(tanh_profile));
  EXPECT_TRUE(HasTrops(tanh_profile));
}

XLA_TEST_F(HloProfileTest, ProfileWhileComputation) {
  const int64 size = 256;
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {size, size});
  Shape while_result_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {}), matrix_shape});

  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetDefaultPlatform());
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client,
                          ClientLibrary::GetOrCreateLocalClient(platform));

  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto state = Parameter(&builder, 0, while_result_shape, "state");
    auto iteration = GetTupleElement(state, 0);
    Gt(ConstantR0<int32>(&builder, 5), iteration);
    TF_ASSERT_OK_AND_ASSIGN(condition, builder.Build());
  }

  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto state = Parameter(&builder, 0, while_result_shape, "state");
    auto matrix = GetTupleElement(state, 1);
    auto next_iteration =
        Add(GetTupleElement(state, 0), ConstantR0<int32>(&builder, 1));
    Tuple(&builder, {next_iteration, Mul(matrix, matrix)});
    TF_ASSERT_OK_AND_ASSIGN(body, builder.Build());
  }

  XlaBuilder builder(TestName());
  auto initial_while_state =
      Tuple(&builder, {ConstantR0<int32>(&builder, 0),
                       Parameter(&builder, 0, matrix_shape, "initial_value")});
  auto while_result = While(condition, body, initial_while_state);
  Add(GetTupleElement(while_result, 1),
      Parameter(&builder, 1, matrix_shape, "other_value"));

  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());

  string profile_output;
  ExecuteAndFetchProfile(&profile_output, client, computation, matrix_shape,
                         matrix_shape);

  std::vector<string> profile_output_lines =
      tensorflow::str_util::Split(profile_output, '\n');

  auto while_body_profile_start =
      c_find_if(profile_output_lines, [](tensorflow::StringPiece s) {
        return tensorflow::str_util::StartsWith(s,
                                                "Execution profile for body");
      });

  ASSERT_NE(while_body_profile_start, profile_output_lines.cend());

  auto while_body_profile_end =
      std::find_if(while_body_profile_start, profile_output_lines.end(),
                   [](tensorflow::StringPiece s) {
                     return tensorflow::str_util::StartsWith(
                         s, "********** microseconds report **********");
                   });

  // We emit a blank line before the "********** microseconds report **********"
  // line.
  while_body_profile_end--;

  ASSERT_NE(while_body_profile_end, profile_output_lines.end());

  gtl::FlatMap<string, ParsedProfileOutputLine> parsed_profile_lines;

  for (auto while_body_profile_i = while_body_profile_start + 1;
       while_body_profile_i != while_body_profile_end; while_body_profile_i++) {
    // There are multiple "get-tuple-element" instructions in the while body so
    // we ignore them -- we don't want parsed_profile_lines to be a multi-map.
    TF_ASSERT_OK(ParseOneProfileOutputLine(
        *while_body_profile_i,
        /*expect_hlo=*/while_body_profile_i != (while_body_profile_start + 1),
        &parsed_profile_lines, {"get-tuple-element"}));
  }

  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine total_while_body_profile,
                          MaybeFind(parsed_profile_lines, "[total]"));
  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine multiply_profile,
                          MaybeFind(parsed_profile_lines, "multiply"));

  EXPECT_GT(total_while_body_profile.cycles, 0);
  EXPECT_EQ(total_while_body_profile.opcode, "[total]");
  EXPECT_EQ(total_while_body_profile.cycles_percentage, "100.00%");

  EXPECT_GT(total_while_body_profile.cycles, multiply_profile.cycles);
  EXPECT_NE(multiply_profile.cycles_percentage, "0.00%");
  EXPECT_NE(multiply_profile.cycles_percentage, "100.00%");
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
  // executable, so block it here. Also block the WhileLoopInvariantCodeMotion
  // pass, otherwise a while loop is transformed and we could not match the
  // original name in the ProfileWhileComputation test.
  new_argv[argc + 1] = strdup(
      "--xla_disable_hlo_passes=fusion,while-loop-invariant-code-motion");
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
