/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/base/casts.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {
class ExhaustiveF32ElementwiseOpTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<std::pair<int64, int64>> {
 protected:
  ErrorSpec error_spec_{0.0001, 0.0001};

  template <typename EnqueueOpTy>
  void ExhaustivelyTestF32Op(EnqueueOpTy enqueue_op,
                             float (*evaluate_op)(float),
                             std::pair<int64, int64> known_incorrect_range) {
    SetFastMathDisabled(true);

    int64 begin, end;
    std::tie(begin, end) = GetParam();
    int64 input_size = end - begin;

    if (begin >= known_incorrect_range.first &&
        end <= known_incorrect_range.second) {
      LOG(INFO) << absl::StreamFormat(
          "Skipping this shard, as the range under test, [%d, %d), falls "
          "entirely within the known-incorrect range [%d, %d).",
          begin, end, known_incorrect_range.first,
          known_incorrect_range.second);
      return;
    }

    LOG(INFO) << "Checking range [" << begin << ", " << end << ")";

    XlaBuilder builder(TestName());

    auto ith_input_elem = [&](int64 i) -> float {
      i += begin;
      // If the operation is known to be buggy on a specific input clamp that
      // input to 0 under the assumption that the op is at least correct on 0.
      if (i >= known_incorrect_range.first &&
          i < known_incorrect_range.second) {
        return 0;
      }
      return absl::bit_cast<float, int32>(i);
    };

    Literal input_literal =
        LiteralUtil::CreateFromDimensions(F32, {input_size});
    absl::Span<float> input_arr = input_literal.data<float>();
    for (int64 i = 0; i < input_size; i++) {
      input_arr[i] = ith_input_elem(i);
    }
    auto input = Parameter(&builder, 0, input_literal.shape(), "input");
    enqueue_op(&builder, input);
    TF_ASSERT_OK_AND_ASSIGN(XlaComputation comp, builder.Build());

    // Build and run the computation using the LocalClient API, rather than the
    // plain Client API, which is used by ClientLibraryTestBase.  This is
    // because the plain Client API results does more memcpys to/from Literals,
    // and that's slow given that we're touching a lot of data here.
    //
    // Copy debug options from ClientLibraryTestBase.  In particular, we're
    // interested in disabling constant folding.
    ExecutableBuildOptions build_opts;
    *build_opts.mutable_debug_options() = *mutable_debug_options();
    TF_ASSERT_OK_AND_ASSIGN(
        auto executable,
        client_->Compile(comp, {&input_literal.shape()}, build_opts));

    TF_ASSERT_OK_AND_ASSIGN(
        ScopedShapedBuffer input_data,
        client_->LiteralToShapedBuffer(input_literal, /*device_ordinal=*/0));

    ExecutableRunOptions run_opts;
    run_opts.set_allocator(client_->backend().memory_allocator());
    run_opts.set_intra_op_thread_pool(
        client_->backend().eigen_intra_op_thread_pool_device());
    TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer result,
                            executable->Run({&input_data}, run_opts));

    TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                            client_->ShapedBufferToLiteral(result));

    // We essentially reimplement LiteralTestUtil::Near here because
    //  a) this streamlined implementation is much faster, and
    //  b) we can print out better error messages (namely, we can print out
    //     which floating-point value input failed, while LiteralTestUtil::Near
    //     can only print out the input index that failed).
    absl::Span<float> result_arr = result_literal.data<float>();
    ASSERT_EQ(result_arr.size(), input_arr.size());
    int64 mismatches = 0;
    for (int64 i = 0; i < input_arr.size(); ++i) {
      float input = ith_input_elem(i);
      float expected = evaluate_op(input);
      float actual = result_arr[i];
      float abs_err = std::abs(expected - actual);
      float rel_err = abs_err / std::abs(expected);
      if (abs_err < error_spec_.abs || rel_err < error_spec_.rel ||
          (std::isnan(expected) && std::isnan(actual)) ||
          (std::isinf(expected) && std::isinf(actual) &&
           (expected > 0) == (actual > 0))) {
        // Successful match!  Nothing to do.
      } else {
        constexpr int64 kMaxMismatchesPrinted = 1000;
        mismatches++;
        if (mismatches < kMaxMismatchesPrinted || VLOG_IS_ON(2)) {
          // Use %0.9g because that's guaranteed to print an f32 to full
          // precision.
          LOG(ERROR) << absl::StreamFormat(
              "Mismatch on %0.9g (0x%08x). Expected %0.9g (0x%08x), but got "
              "%0.9g (0x%08x).",
              input, absl::bit_cast<uint32>(input),        //
              expected, absl::bit_cast<uint32>(expected),  //
              actual, absl::bit_cast<uint32>(actual));
        }
        if (mismatches == kMaxMismatchesPrinted && !VLOG_IS_ON(2)) {
          LOG(ERROR) << "Not printing any more mismatches; pass "
                        "--vmodule=exhaustive_f32_elementwise_op_test=2 to see "
                        "all of them.";
        }
      }
    }
    EXPECT_EQ(mismatches, 0);
  }
};

XLA_TEST_P(ExhaustiveF32ElementwiseOpTest, LogF32) {
#ifdef XLA_TEST_BACKEND_CPU
  // TODO(b/73141998): The vectorized Log implementation gives results outside
  // our error spec in this range (these numbers are bitwise representations of
  // floats expressed as a zero extended int64).
  std::pair<int64, int64> known_incorrect_range = {1, 8388608};
#else
  std::pair<int64, int64> known_incorrect_range = {0, 0};
#endif

  ExhaustivelyTestF32Op(
      [](XlaBuilder* builder, const XlaOp& input) { Log(input); }, std::log,
      known_incorrect_range);
}

XLA_TEST_P(ExhaustiveF32ElementwiseOpTest, ExpF32) {
#ifdef XLA_TEST_BACKEND_CPU
  // TODO(b/73142289): The vectorized Exp implementation gives results outside
  // our error spec in this range (these numbers are bitwise representations of
  // floats expressed as a zero extended int64):
  std::pair<int64, int64> known_incorrect_range = {1107296256 + 11583654,
                                                   1107296256 + 11629080};
#else
  std::pair<int64, int64> known_incorrect_range = {0, 0};
#endif

  ExhaustivelyTestF32Op(
      [](XlaBuilder* builder, const XlaOp& input) { Exp(input); }, std::exp,
      known_incorrect_range);
}

XLA_TEST_P(ExhaustiveF32ElementwiseOpTest, TanhF32) {
  ExhaustivelyTestF32Op(
      [](XlaBuilder* builder, const XlaOp& input) { Tanh(input); }, std::tanh,
      /*known_incorrect_range=*/{0, 0});
}

std::vector<std::pair<int64, int64>> CreateExhaustiveParameters() {
  // We break up the 2^32-element space into small'ish chunks to keep peak
  // memory usage low.
  std::vector<std::pair<int64, int64>> result;
  const int64 step = 1 << 25;
  for (int64 i = 0; i < (1l << 32); i += step) {
    result.push_back({i, i + step});
  }
  return result;
}

INSTANTIATE_TEST_CASE_P(ExhaustiveF32ElementwiseOpTestInstance,
                        ExhaustiveF32ElementwiseOpTest,
                        ::testing::ValuesIn(CreateExhaustiveParameters()));
}  // namespace
}  // namespace xla
