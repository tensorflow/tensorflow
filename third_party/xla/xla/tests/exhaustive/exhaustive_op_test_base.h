/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_BASE_H_
#define XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_BASE_H_

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/bit_cast.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/exhaustive/error_spec.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace exhaustive_op_test {

// Return if the user specified dumping all tested values with their expected
// and actual results.
bool ShouldDumpValues();

// Add all extra CLI flags that are used by ExhaustiveOpTestBase.
void AddExhaustiveFlags(std::vector<tsl::Flag>& flag_list);

// Base class from which all exhaustive tests should inherit.
//
// Holds a bunch of utility functions to simplify the process of running the
// operation and checking against expectations across multiple values.
//
// Type Parameters:
// - T: The primitive type being tested.
// - N: The number of operands that the function being tested takes.
//
// Pure Virtual Functions:
// - GetInputSize
// - FillInput
// - RelaxedDenormalSigns
template <PrimitiveType T, size_t N>
class ExhaustiveOpTestBase : public ClientLibraryTestBase {
 public:
  using Traits = ExhaustiveOpTestTraits<T, N>;
  static constexpr PrimitiveType kT = Traits::kT;

  using NativeT = typename Traits::NativeT;
  using NativeRefT = typename Traits::NativeRefT;
  using ComponentNativeT = typename Traits::ComponentNativeT;
  using ComponentNativeRefT = typename Traits::ComponentNativeRefT;
  using ComponentIntegralNativeT = typename Traits::ComponentIntegralNativeT;
  using ComponentIntegralNativeRefT =
      typename Traits::ComponentIntegralNativeRefT;

  using NativeInputs = typename Traits::NativeInputs;
  using NativeListInputs = typename Traits::NativeListInputs;
  using NativeRefInputs = typename Traits::NativeRefInputs;
  using LiteralInputs = typename Traits::LiteralInputs;
  using XlaInputs = typename Traits::XlaInputs;

  using EvaluateOp = typename Traits::EvaluateOp;
  using EnqueueOp = typename Traits::EnqueueOp;
  using OutputRangeCheck = typename Traits::OutputRangeCheck;
  using ErrorSpecGen = typename Traits::ErrorSpecGen;

  ExhaustiveOpTestBase()
      : should_dump_values_(xla::exhaustive_op_test::ShouldDumpValues()) {
    SetFastMathDisabled(true);

    // Run all HLO passes.  In particular, constant folding is disabled by
    // default for tests, but we need to run it in order to tickle some bugs.
    mutable_debug_options()->clear_xla_disable_hlo_passes();
  }

  // Converts part or all bits in an uint64_t to the value of the floating point
  // data type being tested.
  //
  // When trying to exhaustive test for an operation of data type T, we always
  // use an integral I with the same number of bits at T to exhaustive the input
  // bit patterns for T. This bit pattern is zero extended and stored as
  // uint64_t. This function is used to convert such a bit pattern stored as
  // uint64_t to the input value for T.
  static ComponentNativeT ConvertValue(uint64_t bits) {
    auto used_bits = static_cast<ComponentIntegralNativeT>(bits);
    return BitCast<ComponentNativeT>(used_bits);
  }

  // Returns the number of elements in each input literal.
  virtual int64_t GetInputSize() = 0;

  // Fills the literals with values to test for.
  virtual void FillInput(LiteralInputs* literals) = 0;

  // If true, allows denormals to be flushed to non-sign-preserving 0.
  //
  // For example, normally we'd expect sqrt(-denormal) to be either nan (sqrt of
  // a negative number) or -inf (flush the denormal to sign-preserving zero,
  // then sqrt(-0)). When true, we'll also accept 0 (sqrt(0)).
  //
  // XLA:GPU preserves denormal signs, but other backends don't.
  virtual bool RelaxedDenormalSigns() const = 0;

  // Enable debug logging for the invocation of the lambda.
  //
  // This is intended to be used to wrap a call to `Run`, which will then
  // log extra debug information for a failure such as the calculated
  // absolute, relative, and distance errors. In addition, in an effort to
  // reduce output log size, this will trigger an ASSERT failure to early
  // return from a test at the first failure.
  template <typename Callable,
            std::enable_if_t<std::is_invocable_r_v<void, Callable>, int> = 0>
  void EnableDebugLoggingForScope(Callable&& work) {
    should_emit_debug_logging_ = true;
    work();
    should_emit_debug_logging_ = false;
  }

  // A helper for implementing the Run method for exhaustive op tests. It
  // constructs the HLO module, compiles and runs the module and checks the
  // result.
  //
  // We use a function pointer for evaluate_op for performance because it is
  // called each time an output element is compared inside a loop in routine
  // ExpectNear.
  void Run(EnqueueOp enqueue_op, EvaluateOp evaluate_op,
           OutputRangeCheck check_valid_range = nullptr) {
    Run(enqueue_op, evaluate_op, GetDefaultSpecGenerator<T, N>(),
        check_valid_range);
  }
  void Run(EnqueueOp enqueue_op, EvaluateOp evaluate_op,
           ErrorSpecGen error_spec_gen,
           OutputRangeCheck check_valid_range = nullptr);

  // Builds and runs the computation using the LocalClient API, rather than the
  // plain Client API, which is used by ClientLibraryTestBase.  This is because
  // the plain Client API results does more memcpys to/from Literals, and that's
  // slow given that we're touching a lot of data here.
  absl::StatusOr<Literal> RunComputationHelper(const XlaComputation& comp,
                                               const Literal& literal) {
    return RunComputation(comp, {&literal});
  }
  absl::StatusOr<Literal> RunComputationHelper(
      const XlaComputation& comp, const std::array<Literal, N>& literals) {
    std::array<const Literal*, N> lit_ptrs;
    for (int i = 0; i < N; ++i) {
      lit_ptrs[i] = &literals[i];
    }
    return RunComputation(comp, lit_ptrs);
  }
  absl::StatusOr<Literal> RunComputation(
      const XlaComputation& computation,
      absl::Span<const Literal* const> input_literals);

  // Determines if two output values are sufficiently close to each other based
  // on an error spec.
  bool IsClose(NativeRefT expected, NativeRefT actual, ErrorSpec spec);

  // Returns a list of inputs that should be tested for closeness given some
  // original input values.
  //
  // For denormal component inputs, we accept answers that are close to any of:
  //
  //   - evaluate_op(input)
  //   - evaluate_op(+/-0), where the sign of 0 equal to the sign of
  //     `input`,
  //   - evaluate_op(+/-min_normal_float), where the sign of
  //     min_normal_float matches `input`.
  //   - if relaxed_denormal_signs_, evaluate_op(-/+0), where the sign of
  //     0 is the opposite of `input`.
  //
  // (In particular, the XLA:CPU implementation of log flushes positive
  // denormals to min-normal-float.  This seems kind of reasonable if our
  // goal is to avoid infinities because they cause nans?)
  //
  // For complex numbers, we can find the subnormal testing values for each
  // component, then take the Cartesian product of each set of component values.
  std::vector<ComponentNativeRefT> GetTestValuesWithSubnormalSubstitutions(
      ComponentNativeRefT value);
  std::vector<std::complex<ComponentNativeRefT>>
  GetTestValuesWithSubnormalSubstitutions(
      std::complex<ComponentNativeRefT> value);

  // The test values for an XLA function with N operands are the Cartesian
  // product of the test values for each of the N operands.
  std::vector<NativeRefInputs> GetTestValuesWithSubnormalSubstitutionsArray(
      const NativeRefInputs& value);

  // We essentially reimplement LiteralTestUtil::Near here because
  //  a) this streamlined implementation is much faster, and
  //  b) we can print out better error messages (namely, we can print out
  //     which floating-point value input failed, while
  //     LiteralTestUtil::Near can only print out the input index that
  //     failed).
  //  c) we need special handling of certain inputs.  For example, we say
  //  that
  //     a denormal input has multiple correct outputs (namely, f(x) and
  //     f(0)) and just needs to be close to one of them.
  // check_valid_range can be used to provide a function that is called with
  // the result to check whether it is in the expected range.
  void ExpectNear(const LiteralInputs& input_literals,
                  const Literal& result_literal, EvaluateOp evaluate_op,
                  ErrorSpecGen error_spec_gen,
                  OutputRangeCheck check_valid_range = nullptr);

 protected:
  // Indicates if files of the expected and actual values should be dumped.
  bool should_dump_values_ = false;

  // Indicates if additional (potentially costly) logging should be emitted to
  // ease with debugging.
  bool should_emit_debug_logging_ = false;
};

}  // namespace exhaustive_op_test
}  // namespace xla

#endif  // XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_BASE_H_
