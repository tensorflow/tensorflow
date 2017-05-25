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

#include <initializer_list>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class CompilationCacheTest : public ClientLibraryTestBase {
 public:
  void ExecuteComputationR0F32(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments, float expected_result,
      bool expect_cache_hit) {
    ExecutionProfile execution_profile;
    std::unique_ptr<Literal> result =
        client_
            ->ExecuteAndTransfer(computation, arguments,
                                 /*execution_options=*/nullptr,
                                 &execution_profile)
            .ConsumeValueOrDie();
    LiteralTestUtil::ExpectNear(*LiteralUtil::CreateR0<float>(expected_result),
                                *result, error_spec_);
    EXPECT_EQ(expect_cache_hit, execution_profile.compilation_cache_hit());
  }

  void ExecuteComputationR2F32(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      std::initializer_list<std::initializer_list<float>> expected_result,
      bool expect_cache_hit) {
    ExecutionProfile execution_profile;
    auto data_handle =
        client_
            ->Execute(computation, arguments, /*execution_options=*/nullptr,
                      &execution_profile)
            .ConsumeValueOrDie();
    std::unique_ptr<Literal> result =
        client_->Transfer(*data_handle).ConsumeValueOrDie();
    LiteralTestUtil::ExpectNear(*LiteralUtil::CreateR2<float>(expected_result),
                                *result, error_spec_);
    EXPECT_EQ(expect_cache_hit, execution_profile.compilation_cache_hit());
  }

  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(CompilationCacheTest, ComputationCalledMultipleTimes) {
  ComputationBuilder builder(client_, TestName());
  builder.Neg(builder.ConstantR0<float>(42.0));
  Computation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/true);
  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/true);
}

XLA_TEST_F(CompilationCacheTest, ComputationCalledWithDifferentParameters) {
  std::unique_ptr<GlobalData> data_42 =
      client_->TransferToServer(*LiteralUtil::CreateR0<float>(42.0f))
          .ConsumeValueOrDie();
  std::unique_ptr<GlobalData> data_123 =
      client_->TransferToServer(*LiteralUtil::CreateR0<float>(123.0f))
          .ConsumeValueOrDie();
  std::unique_ptr<GlobalData> data_456 =
      client_->TransferToServer(*LiteralUtil::CreateR0<float>(456.0f))
          .ConsumeValueOrDie();

  ComputationBuilder builder(client_, TestName());
  builder.Neg(builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "param"));
  Computation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation, {data_42.get()}, -42.0,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation, {data_123.get()}, -123.0,
                          /*expect_cache_hit=*/true);
  ExecuteComputationR0F32(computation, {data_456.get()}, -456.0,
                          /*expect_cache_hit=*/true);
  ExecuteComputationR0F32(computation, {data_42.get()}, -42.0,
                          /*expect_cache_hit=*/true);
}

XLA_TEST_F(CompilationCacheTest, MultipleComputations) {
  ComputationBuilder builder_neg(client_, TestName() + "_neg");
  builder_neg.Neg(builder_neg.ConstantR0<float>(42.0));
  Computation computation_neg = builder_neg.Build().ConsumeValueOrDie();

  ComputationBuilder builder_exp(client_, TestName() + "_exp");
  builder_exp.Exp(builder_exp.ConstantR0<float>(1.0));
  Computation computation_exp = builder_exp.Build().ConsumeValueOrDie();

  ComputationBuilder builder_add(client_, TestName() + "_add");
  builder_add.Add(builder_add.ConstantR0<float>(2.0),
                  builder_add.ConstantR0<float>(3.0));
  Computation computation_add = builder_add.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation_neg, {}, -42.0,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation_exp, {}, 2.7182817,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation_add, {}, 5.0,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation_neg, {}, -42.0,
                          /*expect_cache_hit=*/true);
}

XLA_TEST_F(CompilationCacheTest, DifferentParameterLayouts) {
  // Create two GlobalData arrays with the same shape but different
  // layouts. Use these arrays as parameters to a simple computation. If the
  // layout of the array changes then computation should be recompiled (cache
  // miss).
  auto rowmaj_array = test_utils::CreateR2LiteralWithLayout(
      {{1.0f, 2.0f}, {3.0f, 4.0f}}, /*minor_to_major=*/{1, 0});
  auto rowmaj_handle =
      client_->TransferToServer(*rowmaj_array).ConsumeValueOrDie();

  auto colmaj_array = test_utils::CreateR2LiteralWithLayout(
      {{1.0f, 2.0f}, {3.0f, 4.0f}}, /*minor_to_major=*/{0, 1});
  auto colmaj_handle =
      client_->TransferToServer(*colmaj_array).ConsumeValueOrDie();

  ComputationBuilder builder(client_, TestName());
  builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2}), "param0");
  Computation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR2F32(computation, {colmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/false);
  ExecuteComputationR2F32(computation, {colmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/true);
  ExecuteComputationR2F32(computation, {rowmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/false);
  ExecuteComputationR2F32(computation, {rowmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/true);
  ExecuteComputationR2F32(computation, {colmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/true);
}

XLA_TEST_F(CompilationCacheTest, MutatedComputation) {
  // Build a computation, execute it, then mutate it. The mutated computation
  // should not be in the cache until it is run once. This must be done through
  // the stub interface because Computations built from ComputationBuilder are
  // immutable.
  ComputationBuilder builder(client_, TestName());
  auto neg = builder.Neg(builder.ConstantR0<float>(42.0));
  Computation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/true);

  BinaryOpRequest request;
  request.set_binop(BINOP_ADD);
  *request.mutable_lhs() = neg;
  *request.mutable_rhs() = neg;
  OpRequest op_request;
  *op_request.mutable_computation() = computation.handle();
  *op_request.mutable_binary_op_request() = request;
  OpResponse response;
  tensorflow::Status s = client_->stub()->Op(&op_request, &response);
  ASSERT_TRUE(s.ok());

  ExecuteComputationR0F32(computation, {}, -84.0, /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation, {}, -84.0, /*expect_cache_hit=*/true);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
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
