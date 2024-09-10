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
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/bit_cast.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/executable_run_options.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/exhaustive/error_spec.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace exhaustive_op_test {

// Access this through GetEupVersion.
extern int eup_version;

// Get the TPU EUP version (if it was provided).
int GetEupVersion();

// Return if the user specified dumping all tested values with their expected
// and actual results.
bool ShouldDumpValues();

void AddExhaustiveFlags(std::vector<tsl::Flag>& flag_list);

// Get the floating point distance (number of floating point values between)
// expected and actual.
//
// This is a wrapper around xla::CalculateDistanceInFloats for most types. For
// complex types, this returns the maximum distance between the real and
// imaginary components.
template <typename NativeT>
int64_t GetDistanceErr(NativeT expected, NativeT actual) {
  if constexpr (std::is_same_v<NativeT, xla::complex64> ||
                std::is_same_v<NativeT, xla::complex128>) {
    return std::max(
        CalculateDistanceInFloats(expected.real(), actual.real()),
        CalculateDistanceInFloats(expected.imag(), expected.imag()));
  } else {
    return CalculateDistanceInFloats(expected, actual);
  }
}

// Base class from which all exhaustive tests should inherit.
//
// Holds a bunch of utility functions to simplify the process of running the
// operation and checking against expectations across multiple values.
//
// Type Parameters:
// - T: The primitive type being tested.
// - N: The number of operands that the function being tested takes.
template <PrimitiveType T, size_t N>
class ExhaustiveOpTestBase : public ClientLibraryTestBase {
 public:
  using Traits = ExhaustiveOpTestTraits<T, N>;

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
      : ty_(T),
        platform_(client_->platform()->Name()),
        eup_version_(xla::exhaustive_op_test::GetEupVersion()),
        should_dump_values_(xla::exhaustive_op_test::ShouldDumpValues()) {
    SetFastMathDisabled(true);

    // Run all HLO passes.  In particular, constant folding is disabled by
    // default for tests, but we need to run it in order to tickle some bugs.
    mutable_debug_options()->clear_xla_disable_hlo_passes();
  }

  // Enable debug logging for the invocation of the lambda.
  //
  // This is intended to be used to wrap a call to `Run`, which will then log
  // extra debug information for a failure such as the calculated absolute,
  // relative, and distance errors. In addition, in an effort to reduce output
  // log size, this will trigger an ASSERT failure to early return from a test
  // at the first failure.
  template <typename Callable,
            std::enable_if_t<std::is_invocable_r_v<void, Callable>, int> = 0>
  void EnableDebugLoggingForScope(Callable&& work) {
    should_emit_debug_logging_ = true;
    work();
    should_emit_debug_logging_ = false;
  }

  void Run(EnqueueOp enqueue_op, EvaluateOp evaluate_op,
           OutputRangeCheck check_valid_range = nullptr) {
    Run(enqueue_op, evaluate_op, GetDefaultSpecGenerator<T, N>(),
        check_valid_range);
  }

  // A helper for implementing the Run method for exhaustive op tests. It
  // constructs the HLO module, compiles and runs the module and checks the
  // result.
  //
  // We use a function pointer for evaluate_op for performance because it is
  // called each time an output element is compared inside a loop in routine
  // ExpectNear.
  void Run(EnqueueOp enqueue_op, EvaluateOp evaluate_op,
           ErrorSpecGen error_spec_gen,
           OutputRangeCheck check_valid_range = nullptr) {
    LiteralInputs input_literals = CreateLiteralInputs();
    FillInput(&input_literals);

    XlaBuilder builder(TestName());
    XlaInputs xla_inputs;
    for (int i = 0; i < N; ++i) {
      xla_inputs[i] =
          Parameter(&builder, i, input_literals[i].shape(), "input");
    }
    Traits::BuildFromInputs(xla_inputs, enqueue_op);

    TF_ASSERT_OK_AND_ASSIGN(XlaComputation comp, builder.Build());
    TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                            RunComputationHelper(comp, input_literals));

    ExpectNear(input_literals, result_literal, evaluate_op, error_spec_gen,
               check_valid_range);
  }

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

  // We essentially reimplement LiteralTestUtil::Near here because
  //  a) this streamlined implementation is much faster, and
  //  b) we can print out better error messages (namely, we can print out
  //     which floating-point value input failed, while LiteralTestUtil::Near
  //     can only print out the input index that failed).
  //  c) we need special handling of certain inputs.  For example, we say that
  //     a denormal input has multiple correct outputs (namely, f(x) and f(0))
  //     and just needs to be close to one of them.
  // check_valid_range can be used to provide a function that is called with
  // the result to check whether it is in the expected range.
  void ExpectNear(const LiteralInputs& input_literals,
                  const Literal& result_literal, EvaluateOp evaluate_op,
                  ErrorSpecGen error_spec_gen,
                  OutputRangeCheck check_valid_range = nullptr);

  // Builds and runs the computation using the LocalClient API, rather than the
  // plain Client API, which is used by ClientLibraryTestBase.  This is because
  // the plain Client API results does more memcpys to/from Literals, and that's
  // slow given that we're touching a lot of data here.
  absl::StatusOr<Literal> RunComputation(
      const XlaComputation& computation,
      absl::Span<const Literal* const> input_literals) {
    // Copy debug options from ClientLibraryTestBase.  In particular, we're
    // interested in disabling constant folding.
    ExecutableBuildOptions build_opts;
    *build_opts.mutable_debug_options() = *mutable_debug_options();

    std::vector<ScopedShapedBuffer> input_buffers;
    absl::c_transform(input_literals, std::back_inserter(input_buffers),
                      [&](const Literal* input_literal) {
                        return client_
                            ->LiteralToShapedBuffer(*input_literal,
                                                    /*device_ordinal=*/0)
                            .value();
                      });
    std::vector<const Shape*> input_shapes;
    absl::c_transform(input_buffers, std::back_inserter(input_shapes),
                      [&](const ScopedShapedBuffer& buffer) {
                        return &buffer.on_device_shape();
                      });

    TF_ASSIGN_OR_RETURN(
        auto executables,
        client_->Compile(computation, input_shapes, build_opts));

    std::vector<const ShapedBuffer*> input_buffer_pointers;
    absl::c_transform(
        input_buffers, std::back_inserter(input_buffer_pointers),
        [&](const ScopedShapedBuffer& buffer) { return &buffer; });

    ExecutableRunOptions run_opts;
    run_opts.set_allocator(client_->backend().memory_allocator());
    run_opts.set_intra_op_thread_pool(
        client_->backend().eigen_intra_op_thread_pool_device());
    TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result,
                        executables[0]->Run(input_buffer_pointers, run_opts));

    TF_ASSIGN_OR_RETURN(Literal result_literal,
                        client_->ShapedBufferToLiteral(result));
    return std::move(result_literal);
  }

  const std::string& Platform() { return platform_; }

  bool IsGpu(const std::string& platform) const { return platform == "CUDA"; }
  bool IsCpu(const std::string& platform) const { return platform == "Host"; }
  bool IsTpu(const std::string& platform) const {
    return !IsGpu(platform) && !IsCpu(platform);
  }

  int EupVersion() const { return eup_version_; }
  bool IsPreV5Tpu(const std::string& platform) const {
    return IsTpu(platform) && eup_version_ < 2;
  }
  bool IsPreV6Tpu(const std::string& platform) const {
    return IsTpu(platform) && eup_version_ < 3;
  }

  // Returns the number of elements in each input literal.
  virtual int64_t GetInputSize() = 0;

  // Fills the literals with values to test for.
  virtual void FillInput(LiteralInputs* literals) = 0;

  // Replace infinites with max value to help compute errors.
  static ComponentNativeRefT ReplaceInfWithMax(ComponentNativeRefT value) {
    if (std::isinf(value)) {
      return std::copysign(std::numeric_limits<ComponentNativeRefT>::max(),
                           value);
    }
    return value;
  }

  // Returns true if both components are 0, but their sign bits differ.
  static bool CheckSignedZeroError(ComponentNativeRefT expected,
                                   ComponentNativeRefT actual) {
    return expected == 0 && actual == 0 &&
           std::signbit(expected) != std::signbit(actual);
  }

  // Sets the components to 0 if both are NaNs.
  static void RemoveCorrespondingNaNs(ComponentNativeRefT* expected,
                                      ComponentNativeRefT* actual) {
    if (std::isnan(*expected) && std::isnan(*actual)) {
      *expected = 0;
      *actual = 0;
    }
  }

  // The Implementation of the functions above, except for complex inputs.

  static std::complex<ComponentNativeRefT> ReplaceInfWithMax(
      std::complex<ComponentNativeRefT> value) {
    value.real(ReplaceInfWithMax(value.real()));
    value.imag(ReplaceInfWithMax(value.imag()));
    return value;
  }

  static bool CheckSignedZeroError(std::complex<ComponentNativeRefT> expected,
                                   std::complex<ComponentNativeRefT> actual) {
    return CheckSignedZeroError(expected.real(), actual.real()) ||
           CheckSignedZeroError(expected.imag(), actual.imag());
  }

  static void RemoveCorrespondingNaNs(
      std::complex<ComponentNativeRefT>* expected,
      std::complex<ComponentNativeRefT>* actual) {
    ComponentNativeRefT expected_real = expected->real();
    ComponentNativeRefT expected_imag = expected->imag();
    ComponentNativeRefT actual_real = actual->real();
    ComponentNativeRefT actual_imag = actual->imag();
    RemoveCorrespondingNaNs(&expected_real, &actual_real);
    RemoveCorrespondingNaNs(&expected_imag, &actual_imag);
    expected->real(expected_real);
    expected->imag(expected_imag);
    actual->real(actual_real);
    actual->imag(actual_imag);
  }

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
  std::vector<ComponentNativeRefT> GetTestValuesWithSubnormalSubstitutions(
      ComponentNativeRefT value) {
    std::vector<ComponentNativeRefT> test_values;
    if (std::fpclassify(value) == FP_SUBNORMAL) {
      test_values.reserve(relaxed_denormal_signs_ ? 3 : 2);
      test_values.push_back(std::copysign(0, value));
      test_values.push_back(std::copysign(
          std::numeric_limits<ComponentNativeRefT>::min(), value));
      if (relaxed_denormal_signs_) {
        test_values.push_back(std::copysign(0, -value));
      }
    } else {
      test_values.push_back(value);
    }
    return test_values;
  }

  // Similar to complex numbers, we only need to test the components that are
  // subnormal. We can find the subnormal testing values for each component,
  // then take the Cartesian product of each set of component values.
  std::vector<std::complex<ComponentNativeRefT>>
  GetTestValuesWithSubnormalSubstitutions(
      std::complex<ComponentNativeRefT> value) {
    using complex = std::complex<ComponentNativeRefT>;

    auto real_values = GetTestValuesWithSubnormalSubstitutions(value.real());
    auto imag_values = GetTestValuesWithSubnormalSubstitutions(value.imag());

    std::vector<complex> test_values;
    test_values.reserve(real_values.size() * imag_values.size());
    for (auto real : real_values) {
      for (auto imag : imag_values) {
        test_values.push_back(complex(real, imag));
      }
    }

    return test_values;
  }

  // The test values for an XLA function with N operands are the Cartesian
  // product of the test values for each of the N operands.
  std::vector<std::array<NativeRefT, N>>
  GetTestValuesWithSubnormalSubstitutions(
      const std::array<NativeRefT, N>& value) {
    std::vector<std::array<NativeRefT, N>> test_values;

    std::array<std::vector<NativeRefT>, N> component_test_values;
    int total = 1;
    for (int i = 0; i < N; ++i) {
      component_test_values[i] =
          GetTestValuesWithSubnormalSubstitutions(value[i]);
      if (!component_test_values.empty()) {
        total *= component_test_values[i].size();
      }
    }

    // If total == 1, then value has no subnormal components, so we can just
    // return a vector with value in it.
    if (total == 1) {
      test_values.push_back(value);
      return test_values;
    }

    test_values.reserve(total);

    // Perform a Cartesian product of the vectors in component_test_values.
    // We can calculate this by uniquely mapping each integer from 0 to
    // (total - 1) to a list of component indices. The function that maps an
    // integer z to the index of component j is:
    //    component_index(j) =  (i / NumValues(0, j-1)) % NumValues(j, j)
    // and NumIndices(x, y) is the number of values in the Cartesian product of
    // component_test_values[x], component_test_values[x+1], ...
    // component_test_values[y].
    for (int i = 0; i < total; ++i) {
      int accumulated_num_values = 1;
      std::array<NativeRefT, N> test_value;
      for (int j = 0; j < N; ++j) {
        int num_indices = component_test_values[j].size();
        int component_index = (i / accumulated_num_values) % num_indices;
        test_value[j] = component_test_values[j][component_index];
        accumulated_num_values *= num_indices;
      }
      test_values.push_back(std::move(test_value));
    }
    return test_values;
  }

  LiteralInputs CreateLiteralInputs() {
    LiteralInputs literals;
    for (int i = 0; i < N; ++i) {
      literals[i] = LiteralUtil::CreateFromDimensions(T, {GetInputSize()});
    }
    return std::move(literals);
  }

  // Determines if two output values are sufficiently close to each other based
  // on an error spec.
  bool IsClose(NativeRefT expected, NativeRefT actual, ErrorSpec spec) {
    // When two corresponding values are a NaN, they can be considered to have
    // the same value, so the values are just set to 0.
    RemoveCorrespondingNaNs(&expected, &actual);

    if (spec.strict_signed_zeros) {
      if (CheckSignedZeroError(expected, actual)) {
        return false;
      }
    }

    // Replace Inf with Max when calculating absolute or relative errors. This
    // allows the test to pass when another value are close to Inf and the
    // specified absolute or relative errors are not zero.
    double abs_err =
        std::abs(ReplaceInfWithMax(expected) - ReplaceInfWithMax(actual));
    double rel_err = abs_err / std::abs(ReplaceInfWithMax(expected));
    // N.B.: For sub-32-bit floats, NativeRefT is `float`, so ULP comparisons
    // will be wildly off. We convert back to NativeT for this comparison.
    int64_t distance_err = GetDistanceErr(NativeT(expected), NativeT(actual));

    bool passed = abs_err <= spec.abs_err || rel_err <= spec.rel_err ||
                  distance_err <= spec.distance_err;
    if (should_emit_debug_logging_ && !passed) {
      LOG(INFO) << std::setprecision(
                       std::numeric_limits<ComponentNativeT>::max_digits10)
                << "actual: " << actual << "; expected: " << expected
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << "\n\tabs_err: " << abs_err
                << "; spec.abs_err: " << spec.abs_err
                << "\n\trel_err: " << rel_err
                << "; spec.rel_err: " << spec.rel_err
                << "\n\tdistance_err: " << distance_err
                << "; spec.distance_err: " << spec.distance_err;
    }
    return passed;
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
    using I = ComponentIntegralNativeT;
    I used_bits = static_cast<I>(bits);
    return BitCast<ComponentNativeT>(used_bits);
  }

 protected:
  // The primitive type being tested.
  const PrimitiveType ty_;

  // The platform under test.
  const std::string platform_;

  // Version of the EUP for a TPU target. Only relevant for TPU platforms.
  const int eup_version_;

  // If true, allows denormals to be flushed to non-sign-preserving 0.
  //
  // For example, normally we'd expect sqrt(-denormal) to be either nan (sqrt of
  // a negative number) or -inf (flush the denormal to sign-preserving zero,
  // then sqrt(-0)).  But with this as true, we'll also accept 0 (sqrt(0)).
  //
  // XLA:GPU preserves denormal signs, but other backends don't.
  bool relaxed_denormal_signs_ = platform_ != "CUDA";

  // Indicates if files of the expected and actual values should be dumped.
  bool should_dump_values_ = false;

  // Indicates if additional (potentially costly) logging should be emitted to
  // ease with debugging.
  bool should_emit_debug_logging_ = false;
};

template <PrimitiveType T>
class ExhaustiveUnaryTest : public ExhaustiveOpTestBase<T, 1> {
 public:
  static typename ExhaustiveOpTestTraits<T, 1>::ErrorSpecGen
  GetDefaultSpecGenerator() {
    return exhaustive_op_test::GetDefaultSpecGenerator<T, 1>();
  }
};

template <PrimitiveType T>
class ExhaustiveBinaryTest : public ExhaustiveOpTestBase<T, 2> {
 public:
  static typename ExhaustiveOpTestTraits<T, 2>::ErrorSpecGen
  GetDefaultSpecGenerator() {
    return exhaustive_op_test::GetDefaultSpecGenerator<T, 2>();
  }
};

}  // namespace exhaustive_op_test
}  // namespace xla

#endif  // XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_BASE_H_
