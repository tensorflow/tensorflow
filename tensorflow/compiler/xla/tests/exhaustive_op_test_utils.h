/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_EXHAUSTIVE_OP_TEST_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_EXHAUSTIVE_OP_TEST_UTILS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/bit_cast.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace exhaustive_op_test {

struct ErrorSpec {
  float abs_err;
  float rel_err;

  // If true, will consider -0 not near to +0 and vice versa.  Note that
  // +epsilon may still be considered close to -0, depending on the error
  // spec; this only covers the case when both `expected` and `actual` are
  // equal to 0.
  bool strict_signed_zeros = false;

  ErrorSpec(float a, float r) : abs_err(a), rel_err(r) {}
};

// Representations of the reference function passed in by the user.
template <typename NativeRefT, size_t K>
struct EvaluateOpWrapper {};
template <typename NativeRefT>
struct EvaluateOpWrapper<NativeRefT, 1> {
  using type = NativeRefT (*)(NativeRefT);
};
template <typename NativeRefT>
struct EvaluateOpWrapper<NativeRefT, 2> {
  using type = NativeRefT (*)(NativeRefT, NativeRefT);
};

// Representations of the reference function passed in by the user.
template <typename XlaInputs, size_t K>
struct EnqueueOpWrapper {};
template <typename XlaInputs>
struct EnqueueOpWrapper<XlaInputs, 1> {
  using type = std::function<XlaOp(XlaOp)>;
  static XlaOp BuildFromInputs(XlaInputs inputs, type ty) {
    return ty(inputs[0]);
  }
};
template <typename XlaInputs>
struct EnqueueOpWrapper<XlaInputs, 2> {
  using type = std::function<XlaOp(XlaOp, XlaOp)>;
  static XlaOp BuildFromInputs(XlaInputs inputs, type ty) {
    return ty(inputs[0], inputs[1]);
  }
};

// Representations of the ErrorSpecGen function passed in by the user.
template <PrimitiveType T, size_t K>
struct ErrorSpecGenWrapper {};
template <PrimitiveType T>
struct ErrorSpecGenWrapper<T, 1> {
  using NativeT = typename primitive_util::PrimitiveTypeToNative<T>::type;
  using type = ErrorSpec (*)(NativeT);
};
template <PrimitiveType T>
struct ErrorSpecGenWrapper<T, 2> {
  using NativeT = typename primitive_util::PrimitiveTypeToNative<T>::type;
  using type = ErrorSpec (*)(NativeT, NativeT);
};

template <PrimitiveType T, size_t N>
typename ErrorSpecGenWrapper<T, N>::type GetDefaultSpecGenerator();

// T: The primitive type being tested.
// N: The number of operands that the function being tested takes.
template <PrimitiveType T, size_t N>
class ExhaustiveOpTestBase : public ClientLibraryTestBase {
 public:
  // Definitions depending on the primitive type T.

  static constexpr bool kIsComplex = (T == C128 || T == C64);

  // The primitive type used to compute the reference output.
  struct RefT {
    static constexpr PrimitiveType value = (T == F16 || T == BF16) ? F32 : T;
  };

  // The primitive type of the component of T. If T is not complex, then
  // ComponentT = T.
  struct ComponentT {
    static constexpr PrimitiveType value = !kIsComplex ? T
                                           : T == C128 ? F64
                                           : T == C64  ? F32
                                                       : PRIMITIVE_TYPE_INVALID;
  };

  // Same as ComponentT, but for the RefT primitive type.
  struct ComponentRefT {
    static constexpr PrimitiveType value = !kIsComplex           ? RefT::value
                                           : RefT::value == C128 ? F64
                                           : RefT::value == C64
                                               ? F32
                                               : PRIMITIVE_TYPE_INVALID;
  };

  // The primitive type of an unsigned integer that can be bitcasted to and from
  // ComponentT.
  struct ComponentIntegralT {
    static constexpr PrimitiveType value = (T == C128 || T == F64)  ? U64
                                           : (T == C64 || T == F32) ? U32
                                           : (T == F16 || T == BF16)
                                               ? U16
                                               : PRIMITIVE_TYPE_INVALID;
  };

  // Native types that correspond to the primitive types above.
  using NativeT = typename primitive_util::PrimitiveTypeToNative<T>::type;
  using NativeRefT =
      typename primitive_util::PrimitiveTypeToNative<RefT::value>::type;
  using ComponentNativeT =
      typename primitive_util::PrimitiveTypeToNative<ComponentT::value>::type;
  using ComponentNativeRefT = typename primitive_util::PrimitiveTypeToNative<
      ComponentRefT::value>::type;
  using ComponentIntegralNativeT =
      typename primitive_util::PrimitiveTypeToNative<
          ComponentIntegralT::value>::type;

  using InputLiterals = std::array<Literal, N>;

 private:
  // N spans corresponding to the list of literal data values.
  using NativeInputsList = std::array<absl::Span<const NativeT>, N>;

  // N data items representing a single input to an XLA function.
  using NativeInputs = std::array<NativeT, N>;

  // N data items representing a single input to an interpreter backend
  // function.
  using NativeRefInputs = std::array<NativeRefT, N>;

  // N data items representing a single input to an XLA function.
  using XlaInputs = std::array<XlaOp, N>;

 public:
  using ErrorSpecGen = typename ErrorSpecGenWrapper<T, N>::type;
  using EvaluateOp = typename EvaluateOpWrapper<NativeRefT, N>::type;
  using EnqueueOp = typename EnqueueOpWrapper<XlaInputs, N>::type;

  explicit ExhaustiveOpTestBase()
      : ty_(T), platform_(client_->platform()->Name()) {
    SetFastMathDisabled(true);

    // Run all HLO passes.  In particular, constant folding is disabled by
    // default for tests, but we need to run it in order to tickle some bugs.
    mutable_debug_options()->clear_xla_disable_hlo_passes();
  }

  void Run(EnqueueOp enqueue_op, EvaluateOp evaluate_op) {
    Run(enqueue_op, evaluate_op, GetDefaultSpecGenerator<T, N>());
  }

  // A helper for implementing the Run method for exhaustive op tests. It
  // constructs the HLO module, compiles and runs the module and checks the
  // result.
  //
  // We use a function pointer for evaluate_op for performance because it is
  // called each time an output element is compared inside a loop in routine
  // ExpectNear.
  void Run(EnqueueOp enqueue_op, EvaluateOp evaluate_op,
           ErrorSpecGen error_spec_gen) {
    InputLiterals input_literals = CreateInputLiterals();
    FillInput(&input_literals);

    XlaBuilder builder(TestName());
    XlaInputs xla_inputs;
    for (int i = 0; i < N; ++i) {
      xla_inputs[i] =
          Parameter(&builder, i, input_literals[i].shape(), "input");
    }
    EnqueueOpWrapper<XlaInputs, N>::BuildFromInputs(xla_inputs, enqueue_op);

    TF_ASSERT_OK_AND_ASSIGN(XlaComputation comp, builder.Build());
    TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                            RunComputationHelper(comp, input_literals));
    ExpectNear(input_literals, result_literal, evaluate_op, error_spec_gen);
  }

  StatusOr<Literal> RunComputationHelper(const XlaComputation& comp,
                                         const Literal& literal) {
    return RunComputation(comp, {&literal});
  }

  StatusOr<Literal> RunComputationHelper(
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
  void ExpectNear(const InputLiterals& input_literals,
                  const Literal& result_literal, EvaluateOp evaluate_op,
                  ErrorSpecGen error_spec_gen);

  // Builds and runs the computation using the LocalClient API, rather than the
  // plain Client API, which is used by ClientLibraryTestBase.  This is because
  // the plain Client API results does more memcpys to/from Literals, and that's
  // slow given that we're touching a lot of data here.
  StatusOr<Literal> RunComputation(
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

  // Returns the number of elements in each input literal.
  virtual int64_t GetInputSize() = 0;

  // Fills the literals with values to test for.
  virtual void FillInput(InputLiterals* literals) = 0;

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

  InputLiterals CreateInputLiterals() {
    InputLiterals literals;
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

    return abs_err <= spec.abs_err || rel_err <= spec.rel_err;
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

  ComponentNativeT ConvertAndReplaceKnownIncorrectValueWith(
      uint64_t bits, int replacement_value = 0) {
    if (known_incorrect_fn_ && known_incorrect_fn_(bits)) {
      return static_cast<ComponentNativeT>(replacement_value);
    }
    return ConvertValue(bits);
  }

 protected:
  // The primitive type being tested.
  const PrimitiveType ty_;

  // The platform under test.
  const std::string platform_;

  // Testing will ignore inputs for which known_incorrect_fn_ returns true. The
  // argument to the function is the raw bits for the data being test, zero
  // extended to 64 bits if the data type is less than 64 bits.
  std::function<bool(int64_t)> known_incorrect_fn_;

  // If true, allows denormals to be flushed to non-sign-preserving 0.
  //
  // For example, normally we'd expect sqrt(-denormal) to be either nan (sqrt of
  // a negative number) or -inf (flush the denormal to sign-perserving zero,
  // then sqrt(-0)).  But with this as true, we'll also accept 0 (sqrt(0)).
  //
  // XLA:GPU preserves denormal signs, but other backends don't.
  bool relaxed_denormal_signs_ = platform_ != "CUDA";
};

// Represents a set of 64 bit chunks by representing the starting bit chunk,
// the last bit chunk, and the spacing between two adjacent bit chunks, without
// actually storing all the bit chunks being generated. The bit chunk iterator
// is provided to retrieve all the bit chunks.
//
// This data structure is used to generate the bit representation to test
// operations that requires more than 64 bit input data. In this case,
// truly exhaustive testing is not possible and we want to test a value every
// n values, where n == spacing_.
//
// Currently, the iterator of BitChunks adds the `spacing_` to a bit chunk to
// compute the next bit chunk. We can change this to use values generated
// by a random number generator that can achieve the average spacing
// statistically, if we will find this is necessary.
class BitChunks {
 public:
  class iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = uint64_t;
    using difference_type = uint64_t;
    using pointer = const uint64_t*;
    using reference = uint64_t;

    iterator() = default;

    explicit iterator(const BitChunks* bit_chunks)
        : bit_chunks_(bit_chunks), next_bit_chunk_(bit_chunks->start_) {}

    iterator& operator++() {
      Next();
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      Next();
      return retval;
    }

    bool operator==(iterator other) const {
      return bit_chunks_ == other.bit_chunks_ &&
             next_bit_chunk_ == other.next_bit_chunk_;
    }

    bool operator!=(iterator other) const { return !(*this == other); }

    iterator MoveToEnd() {
      MoveNextBitChunkToOnePassEnd();
      return *this;
    }

    reference operator*() const {
      CHECK(*this != this->bit_chunks_->end());
      return next_bit_chunk_;
    }

    const BitChunks* GetBitChunks() const { return bit_chunks_; }

    void Reset() { next_bit_chunk_ = bit_chunks_->start_; }

    void Next() {
      CHECK(*this != this->bit_chunks_->end());
      if (next_bit_chunk_ == bit_chunks_->end_) {
        MoveNextBitChunkToOnePassEnd();
      } else {
        next_bit_chunk_ += bit_chunks_->spacing_;
        if (next_bit_chunk_ > bit_chunks_->end_) {
          next_bit_chunk_ = bit_chunks_->end_;
        }
      }
    }

    std::string ToString() const {
      return absl::StrFormat("0x%08x", next_bit_chunk_);
    }

   private:
    // Move next_bit_chunk_ to 1 pass the bit_chunks_->end, to mark that the
    // iterator has reached the end. When spacing_ is not one, or if we will
    // change to use a random value instead of spacing_ in function Next(),
    // normalizing the representation of the iterator ending this way can
    // can simplify the checking for iterator ending.
    void MoveNextBitChunkToOnePassEnd() {
      next_bit_chunk_ = bit_chunks_->end_ + 1;
    }

    const BitChunks* bit_chunks_;
    uint64_t next_bit_chunk_;
  };

  iterator begin() const { return iterator(this); }
  iterator end() const {
    iterator end(this);
    return end.MoveToEnd();
  }

  explicit BitChunks(uint64_t start = 0, uint64_t end = 0, uint64_t spacing = 1)
      : start_(start), end_(end), spacing_(spacing) {
    CHECK_GE(end_, start_);
    CHECK_NE(spacing, 0) << ToString();
  }

  int64_t GetTotalBitChunks() const {
    if (start_ == end_) {
      return 1;
    }

    return 1 + (end_ - start_ + spacing_ - 1) / spacing_;
  }

  std::string ToString() const {
    return absl::StrFormat("(0x%08x, 0x%08x, 0x%08x)", start_, end_, spacing_);
  }

  uint64_t start_;
  uint64_t end_;
  uint64_t spacing_;
};

inline std::string StringifyNum(BitChunks c) { return c.ToString(); }

inline std::string StringifyNum(BitChunks::iterator c) { return c.ToString(); }

template <typename T>
void AppendStringifyNum(std::string* s, T x) {
  absl::StrAppend(s, StringifyNum(x));
}

// Represents a set of floating point values through the possible values for
// the three components: mantissa, exponent, and sign. Also implements an
// iterator for retrieving all the represented floating point values.
class FpValues {
 public:
  static constexpr int kTotalBitChunks = 3;

  class iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = uint64_t;
    using difference_type = uint64_t;
    using pointer = const uint64_t*;
    using reference = uint64_t;

    explicit iterator(const FpValues* fp_values) : fp_values_(fp_values) {
      for (int i = 0; i < FpValues::kTotalBitChunks; ++i) {
        iters_[i] = BitChunks::iterator(&fp_values->GetBitChunks(i));
      }
    }

    iterator& operator++() {
      Next();
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      Next();
      return retval;
    }

    bool operator==(iterator other) const {
      for (int i = 0; i < FpValues::kTotalBitChunks; ++i) {
        if (iters_[i] != other.GetBitChunksIter(i)) {
          return false;
        }
      }
      return true;
    }

    bool operator!=(iterator other) const { return !(*this == other); }

    iterator MoveToEnd() {
      for (int i = 0; i < FpValues::kTotalBitChunks; ++i) {
        iters_[i].MoveToEnd();
      }
      return *this;
    }

    uint64_t operator*() const {
      uint64_t value = 0;
      for (int i = 0; i < FpValues::kTotalBitChunks; ++i) {
        value = value | (*iters_[i]) << fp_values_->offsets_[i];
      }
      return value;
    }

    const BitChunks::iterator& GetBitChunksIter(int i) { return iters_[i]; }

    std::string ToString() const {
      return absl::StrJoin(iters_, ",",
                           AppendStringifyNum<BitChunks::iterator>);
    }

   private:
    // Moves the iterator for the ith BitChunks to the next value, and
    // returns true if the new state is not the end of the iterator.
    bool Next(int i = 0) {
      iters_[i].Next();
      if (iters_[i] == iters_[i].GetBitChunks()->end()) {
        if (i == FpValues::kTotalBitChunks - 1) {
          return false;
        }
        if (Next(i + 1)) {
          iters_[i].Reset();
          return true;
        }
        return false;
      }
      return true;
    }

    std::array<BitChunks::iterator, FpValues::kTotalBitChunks> iters_;
    const FpValues* fp_values_;
  };

  FpValues() : bit_chunks_(), offsets_() {}
  FpValues(absl::Span<const BitChunks> chunks, absl::Span<const int> offsets) {
    CHECK_EQ(chunks.size(), offsets.size() - 1);
    CHECK_EQ(chunks.size(), kTotalBitChunks);
    std::copy_n(chunks.begin(), kTotalBitChunks, bit_chunks_.begin());
    std::copy_n(offsets.begin(), kTotalBitChunks, offsets_.begin());

    // The last value in `offsets` is the total number of bits.
    offsets_[kTotalBitChunks] = offsets[kTotalBitChunks];
    // Validate the input values.
    for (int i = 0; i < kTotalBitChunks; ++i) {
      int total_bits = offsets[i + 1] - offsets[i];
      if (total_bits < 64) {
        uint64_t bound = 1ull << total_bits;
        CHECK_LT(chunks[i].start_, bound);
        CHECK_LT(chunks[i].end_, bound);
      } else {
        CHECK_EQ(total_bits, 64);
      }
    }
  }

  iterator begin() const { return iterator(this); }

  iterator end() const {
    iterator end(this);
    return end.MoveToEnd();
  }

  int64_t GetTotalNumValues() const {
    int64_t total = 1;
    absl::c_for_each(bit_chunks_, [&](const BitChunks& chunks) {
      total *= chunks.GetTotalBitChunks();
    });
    return total;
  }

  const BitChunks& GetBitChunks(int i) const { return bit_chunks_[i]; }

  std::string ToString() const {
    return absl::StrCat(
        "[", absl::StrJoin(bit_chunks_, ",", AppendStringifyNum<BitChunks>),
        "]");
  }

  std::array<BitChunks, kTotalBitChunks> bit_chunks_;
  std::array<int, kTotalBitChunks + 1> offsets_;
};

template <typename T, typename std::enable_if<
                          std::is_same<T, float>::value ||
                          std::is_same<T, double>::value>::type* = nullptr>
int GetMantissaTotalBits() {
  return std::numeric_limits<T>::digits - 1;
}

template <typename T>
int GetFpTotalBits() {
  return sizeof(T) * 8;
}

template <typename T>
int GetExponentTotalBits() {
  return GetFpTotalBits<T>() - GetMantissaTotalBits<T>() - 1;
}

template <typename T>
uint64_t GetAllOneMantissa() {
  return (1ull << GetMantissaTotalBits<T>()) - 1ull;
}

template <typename T>
uint64_t GetAllOneExponent() {
  return (1ull << GetExponentTotalBits<T>()) - 1ull;
}

template <typename T, typename std::enable_if<
                          std::is_same<T, float>::value ||
                          std::is_same<T, double>::value>::type* = nullptr>
FpValues GetFpValues(BitChunks mantissa, BitChunks exponent, BitChunks sign) {
  int total_bits = GetFpTotalBits<T>();
  return FpValues({mantissa, exponent, sign},
                  {0, GetMantissaTotalBits<T>(), total_bits - 1, total_bits});
}

template <typename T>
FpValues GetZeros() {
  return GetFpValues<T>(BitChunks(0, 0, 1), BitChunks(0, 0, 1),
                        BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetSubnormals(int approx_num_values) {
  int mantissa = GetMantissaTotalBits<T>();
  uint64_t mantissa_spacing = (1ull << mantissa) / (approx_num_values * 2);
  return GetFpValues<T>(
      BitChunks(0x1, GetAllOneMantissa<T>(), mantissa_spacing),
      BitChunks(0, 0, 1), BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetInfinites() {
  uint64_t all_one_exp = GetAllOneExponent<T>();
  return GetFpValues<T>(BitChunks(0, 0, 1),
                        BitChunks(all_one_exp, all_one_exp, 1),
                        BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetNans(int approx_num_values) {
  int mantissa = GetMantissaTotalBits<T>();
  uint64_t mantissa_spacing = (1ull << mantissa) / (approx_num_values * 2);
  uint64_t all_one_exp = GetAllOneExponent<T>();
  return GetFpValues<T>(
      BitChunks(0x1, GetAllOneMantissa<T>(), mantissa_spacing),
      BitChunks(all_one_exp, all_one_exp, 1), BitChunks(0, 1, 1));
}

template <typename T>
FpValues GetNormals(int approx_num_values) {
  float component_total = std::sqrt(static_cast<float>(approx_num_values));
  return GetFpValues<T>(
      BitChunks(0x1, GetAllOneMantissa<T>(),
                (1ull << (GetMantissaTotalBits<T>() + 1)) / component_total),
      BitChunks(0x1, GetAllOneExponent<T>() - 1,
                (1ull << (GetExponentTotalBits<T>() + 1)) / component_total),
      BitChunks(0, 1, 1));
}

// Returns a vector of FpValues, which together represent about
// `approx_num_values` floating point values of type `T`, with each FpValues
// represents about `num_values_per_group` floating point values.
template <typename T>
std::vector<FpValues> GetFpValuesWithExponents(uint64_t first_exponent,
                                               uint64_t exponent_spacing,
                                               uint64_t num_exponents,
                                               uint64_t approx_num_values,
                                               uint64_t num_values_per_group) {
  const uint64_t num_signs = 2;
  uint64_t approx_num_mantissa =
      approx_num_values / (num_exponents * num_signs);
  uint64_t num_mantissa_per_group =
      num_values_per_group / (num_exponents * num_signs);
  CHECK_GT(approx_num_mantissa, 0);
  CHECK_GT(num_mantissa_per_group, 0);

  CHECK_LT(first_exponent + num_exponents - 1ull, GetAllOneExponent<T>());
  int mantissa = GetMantissaTotalBits<T>();
  uint64_t mantissa_spacing = (1ull << mantissa) / approx_num_mantissa;

  std::vector<FpValues> result;
  for (uint64_t group_start = 0; group_start < GetAllOneMantissa<T>();
       group_start += mantissa_spacing * num_mantissa_per_group) {
    uint64_t group_end =
        group_start + (num_mantissa_per_group - 1) * mantissa_spacing;
    if (group_end > GetAllOneMantissa<T>()) {
      group_end = GetAllOneMantissa<T>();
    }
    result.push_back(GetFpValues<T>(
        BitChunks(group_start, group_end, mantissa_spacing),
        BitChunks(first_exponent, first_exponent + num_exponents - 1, 1),
        BitChunks(0, 1, 1)));
  }
  return result;
}

// Returns a vector of FpValues together represent about `approx_num_values`
// "very large" floating point values and `approx_num_values` "very small"
// floating point values of type `T`, which each FpValues represent about
// `num_values_per_group` floating point values. Because we use FpValues as
// a parameter for parameterized testing, the number of floating values
// represented by each FpValues affects the input size for each sub-test and
// the hence the peak memory usage of the test.
template <typename T>
std::vector<FpValues> GetFpValuesForMagnitudeExtremeNormals(
    uint64_t approx_num_values = 40000, uint64_t num_values_per_group = 4000) {
  std::vector<FpValues> large =
      GetFpValuesWithExponents<T>(GetAllOneExponent<T>() - 5, 1, 5,
                                  approx_num_values / 2, num_values_per_group);
  std::vector<FpValues> small = GetFpValuesWithExponents<T>(
      1, 1, 5, approx_num_values / 2, num_values_per_group);
  large.insert(large.end(), small.begin(), small.end());
  return large;
}

template <typename T>
std::vector<FpValues> CreateFpValuesForBoundaryTest() {
  return {GetZeros<T>(), GetSubnormals<T>(1000), GetInfinites<T>(),
          GetNans<T>(1000)};
}

inline std::vector<std::pair<int64_t, int64_t>> CreateExhaustiveF32Ranges() {
  // We break up the 2^32-element space into small'ish chunks to keep peak
  // memory usage low.
  std::vector<std::pair<int64_t, int64_t>> result;
  const int64_t step = 1 << 25;
  for (int64_t i = 0; i < (1l << 32); i += step) {
    result.push_back({i, i + step});
  }
  return result;
}

template <PrimitiveType T, size_t N>
inline ErrorSpec DefaultSpecGenerator(
    typename ExhaustiveOpTestBase<T, N>::NativeT) {
  LOG(FATAL) << "Unhandled Type";
}

template <PrimitiveType T, size_t N>
inline ErrorSpec DefaultSpecGenerator(
    typename ExhaustiveOpTestBase<T, N>::NativeT,
    typename ExhaustiveOpTestBase<T, N>::NativeT) {
  LOG(FATAL) << "Unhandled Type";
}

template <>
inline ErrorSpec DefaultSpecGenerator<C128, 1>(complex128) {
  return ErrorSpec{0.0001, 0.0001};
}

template <>
inline ErrorSpec DefaultSpecGenerator<C64, 1>(complex64) {
  return ErrorSpec{0.0001, 0.0001};
}

template <>
inline ErrorSpec DefaultSpecGenerator<F64, 1>(double) {
  return ErrorSpec{0.0001, 0.0001};
}

template <>
inline ErrorSpec DefaultSpecGenerator<F32, 1>(float) {
  return ErrorSpec{0.0001, 0.0001};
}

template <>
inline ErrorSpec DefaultSpecGenerator<F16, 1>(Eigen::half) {
  return ErrorSpec{0.001, 0.001};
}

template <>
inline ErrorSpec DefaultSpecGenerator<BF16, 1>(bfloat16) {
  return ErrorSpec{0.002, 0.02};
}

template <>
inline ErrorSpec DefaultSpecGenerator<F64, 2>(double, double) {
  return ErrorSpec{0.001, 0.001};
}

template <>
inline ErrorSpec DefaultSpecGenerator<F32, 2>(float, float) {
  return ErrorSpec{0.001, 0.001};
}

template <>
inline ErrorSpec DefaultSpecGenerator<F16, 2>(Eigen::half, Eigen::half) {
  return ErrorSpec{0.001, 0.001};
}

template <>
inline ErrorSpec DefaultSpecGenerator<BF16, 2>(bfloat16, bfloat16) {
  return ErrorSpec{0.002, 0.02};
}

template <PrimitiveType T, size_t N>
typename ErrorSpecGenWrapper<T, N>::type GetDefaultSpecGenerator() {
  return DefaultSpecGenerator<T, N>;
}

template <typename T, typename std::enable_if<
                          std::is_same<T, float>::value ||
                          std::is_same<T, double>::value>::type* = nullptr>
T ReferenceMax(T x, T y) {
  // We need to propagate NAN here because std::max may not propagate NAN.
  if (std::fpclassify(x) == FP_NAN) {
    return x;
  }
  if (std::fpclassify(y) == FP_NAN) {
    return y;
  }

  return std::max<T>(x, y);
}

template <typename T, typename std::enable_if<
                          std::is_same<T, float>::value ||
                          std::is_same<T, double>::value>::type* = nullptr>
T ReferenceMin(T x, T y) {
  // We need to propagate NAN here because std::max may not propagate NAN.
  if (std::fpclassify(x) == FP_NAN) {
    return x;
  }
  if (std::fpclassify(y) == FP_NAN) {
    return y;
  }

  return std::min<T>(x, y);
}

// Returns a wrapper of the given build method, which build an HLO operation
// with an empty broadcast dimension.
inline std::function<XlaOp(XlaOp, XlaOp)> AddEmptyBroadcastDimension(
    std::function<XlaOp(XlaOp, XlaOp, absl::Span<const int64_t>)>
        build_method) {
  return [&](XlaOp src0, XlaOp src1) -> XlaOp {
    return build_method(src0, src1, {});
  };
}

template <PrimitiveType T>
class ExhaustiveUnaryTest : public ExhaustiveOpTestBase<T, 1> {
 public:
  using typename ExhaustiveOpTestBase<T, 1>::ErrorSpecGen;
  static ErrorSpecGen GetDefaultSpecGenerator() {
    return exhaustive_op_test::GetDefaultSpecGenerator<T, 1>();
  }
};

template <PrimitiveType T>
using ExhaustiveBinaryTest = ExhaustiveOpTestBase<T, 2>;

}  // namespace exhaustive_op_test
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_TESTS_EXHAUSTIVE_OP_TEST_UTILS_H_
