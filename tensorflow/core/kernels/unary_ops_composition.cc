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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/relu_op_functor.h"

namespace tensorflow {

template <typename T>
class UnaryOpsComposition;  // forward declare kernel

template <typename T>
struct UnaryOpsCompositionSupport;

template <typename T>
struct UnaryOpsCompositionBase {
  using InputBuffer = typename TTypes<T>::ConstFlat;
  using OutputBuffer = typename TTypes<T>::Flat;

  using ComputeFn = void (*)(const InputBuffer&, OutputBuffer*);

  struct ComputeFnRegistration {
    ComputeFn compute_fn;
    int cost;
  };

  bool HasComputeFn(const string& name) {
    return compute_fns.find(name) != compute_fns.end();
  }

 protected:
  void RegisterComputeFn(const string& name, ComputeFn compute_fn, int cost) {
    VLOG(5) << "Register compute fn: name=" << name << " cost=" << cost;
    compute_fns[name] = {compute_fn, cost};
  }

 private:
  friend class UnaryOpsComposition<T>;

  Status ExportComputeFns(const std::vector<string>& op_names,
                          std::vector<ComputeFn>* fns, int* cost) {
    for (const string& op_name : op_names) {
      auto it = compute_fns.find(op_name);
      if (it == compute_fns.end())
        return errors::InvalidArgument(
            "Do not have a compute function registered for op: ", op_name);

      const ComputeFnRegistration& reg = it->second;
      fns->push_back(reg.compute_fn);
      *cost += reg.cost;
    }

    return OkStatus();
  }

  std::unordered_map<string, ComputeFnRegistration> compute_fns;
};

template <typename T>
class UnaryOpsComposition : public OpKernel {
 public:
  using Kernel = UnaryOpsComposition<T>;

  using Scalar = T;
  using Packet = typename Eigen::internal::packet_traits<T>::type;

  using Support = UnaryOpsCompositionSupport<T>;

  using InputBuffer = typename Support::InputBuffer;
  using OutputBuffer = typename Support::OutputBuffer;
  using ComputeFn = typename Support::ComputeFn;

  explicit UnaryOpsComposition(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("op_names", &op_names_));

    OP_REQUIRES(context, !op_names_.empty(),
                errors::InvalidArgument(
                    "Unary op composition must have at least one op"));

    OP_REQUIRES_OK(context,
                   support_.ExportComputeFns(op_names_, &fns_, &cost_));

    VLOG(2) << "Composed unary op: [" << absl::StrJoin(op_names_, ", ")
            << "]; cost=" << cost_;
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output({0}, 0, in.shape(), &out));

    InputBuffer in_flat = in.flat<T>();
    OutputBuffer out_flat = out->flat<T>();

    const std::size_t num_fns = fns_.size();
    auto compute_fn = [this, &in_flat, &out_flat, &num_fns](int64_t begin,
                                                            int64_t end) {
      int64_t len = end - begin;
      const InputBuffer in_slice(in_flat.data() + begin, len);
      const InputBuffer scratch_slice(out_flat.data() + begin, len);
      OutputBuffer out_slice(out_flat.data() + begin, len);

      fns_[0](in_slice, &out_slice);
      for (int i = 1; i < num_fns; ++i) {
        fns_[i](scratch_slice, &out_slice);
      }
    };

    const CPUDevice& device = ctx->eigen_device<CPUDevice>();
    const int kOverheadCycles = static_cast<int>(num_fns) * 10;
    Eigen::TensorOpCost cost(/*bytes_loaded=*/sizeof(T) * num_fns,
                             /*bytes_stored=*/sizeof(T) * num_fns,
                             kOverheadCycles + cost_);
    device.parallelFor(in.NumElements(), cost, AlignBlockSize,
                       std::move(compute_fn));
  }

 private:
  static constexpr int kPacketSize =
      Eigen::internal::unpacket_traits<Packet>::size;

  static inline int64_t AlignBlockSize(int64_t block_size) {
    // Align block size to packet size and account for unrolling in run above.
    if (block_size >= 16 * kPacketSize) {
      return (block_size + 4 * kPacketSize - 1) & ~(4 * kPacketSize - 1);
    }
    // Aligning to 4 * PacketSize would increase block size by more than 25%.
    return (block_size + kPacketSize - 1) & ~(kPacketSize - 1);
  }

  Support support_;

  std::vector<string> op_names_;
  std::vector<ComputeFn> fns_;
  int cost_ = 0;
};

// Register compute functions for UnaryOp functors.
#define REGISTER_COMPUTE_FN_HELPER(name, functor)                              \
  static_assert(std::is_same<functor::in_type, functor::out_type>::value,      \
                "Functor must have same input and output types");              \
                                                                               \
  static inline void Compute##name(const InputBuffer& in, OutputBuffer* out) { \
    *out = in.unaryExpr(functor::func());                                      \
  }                                                                            \
  static inline int Cost##name() {                                             \
    return Eigen::internal::functor_traits<functor::func>::Cost;               \
  }

// Register compute function for the Relu/Relu6/Elu/Selu.
#define REGISTER_RELU_HELPER()                                                \
  template <typename T>                                                       \
  using functor_traits = Eigen::internal::functor_traits<T>;                  \
                                                                              \
  static inline void ComputeRelu(const InputBuffer& in, OutputBuffer* out) {  \
    auto relu = functor::Relu<Eigen::DefaultDevice, T>();                     \
    relu(Eigen::DefaultDevice(), in, *out);                                   \
  }                                                                           \
                                                                              \
  static inline int CostRelu() {                                              \
    return functor_traits<Eigen::internal::scalar_max_op<T>>::Cost;           \
  }                                                                           \
                                                                              \
  static inline void ComputeRelu6(const InputBuffer& in, OutputBuffer* out) { \
    auto relu6 = functor::Relu6<Eigen::DefaultDevice, T>();                   \
    relu6(Eigen::DefaultDevice(), in, *out);                                  \
  }                                                                           \
                                                                              \
  static inline int CostRelu6() {                                             \
    return functor_traits<Eigen::internal::scalar_max_op<T>>::Cost +          \
           functor_traits<Eigen::internal::scalar_min_op<T>>::Cost;           \
  }                                                                           \
  static inline void ComputeElu(const InputBuffer& in, OutputBuffer* out) {   \
    auto elu = functor::Elu<Eigen::DefaultDevice, T>();                       \
    elu(Eigen::DefaultDevice(), in, *out);                                    \
  }                                                                           \
                                                                              \
  static inline int CostElu() {                                               \
    return functor_traits<Eigen::internal::scalar_exp_op<T>>::Cost +          \
           Eigen::NumTraits<T>::MulCost;                                      \
  }                                                                           \
  static inline void ComputeSelu(const InputBuffer& in, OutputBuffer* out) {  \
    auto selu = functor::Selu<Eigen::DefaultDevice, T>();                     \
    selu(Eigen::DefaultDevice(), in, *out);                                   \
  }                                                                           \
                                                                              \
  static inline int CostSelu() {                                              \
    return 2 * (functor_traits<Eigen::internal::scalar_exp_op<T>>::Cost +     \
                Eigen::NumTraits<T>::MulCost);                                \
  }

#define REGISTER_COMPUTE_FN(func) \
  RegisterComputeFn(#func, Compute##func, Cost##func());

template <>
struct UnaryOpsCompositionSupport<float> : UnaryOpsCompositionBase<float> {
  using T = float;

  UnaryOpsCompositionSupport() {
    // UnaryOp functors.
    REGISTER_COMPUTE_FN(Abs);
    REGISTER_COMPUTE_FN(Acos);
    REGISTER_COMPUTE_FN(Acosh);
    REGISTER_COMPUTE_FN(Asin);
    REGISTER_COMPUTE_FN(Asinh);
    REGISTER_COMPUTE_FN(Atan);
    REGISTER_COMPUTE_FN(Atanh);
    REGISTER_COMPUTE_FN(Ceil);
    REGISTER_COMPUTE_FN(Cos);
    REGISTER_COMPUTE_FN(Cosh);
    REGISTER_COMPUTE_FN(Expm1);
    REGISTER_COMPUTE_FN(Exp);
    REGISTER_COMPUTE_FN(Floor);
    REGISTER_COMPUTE_FN(Inv);
    REGISTER_COMPUTE_FN(Log);
    REGISTER_COMPUTE_FN(Log1p);
    REGISTER_COMPUTE_FN(Neg);
    REGISTER_COMPUTE_FN(Reciprocal);
    REGISTER_COMPUTE_FN(Rint);
    REGISTER_COMPUTE_FN(Round);
    REGISTER_COMPUTE_FN(Rsqrt);
    REGISTER_COMPUTE_FN(Sigmoid);
    REGISTER_COMPUTE_FN(Sin);
    REGISTER_COMPUTE_FN(Sinh);
    REGISTER_COMPUTE_FN(Sqrt);
    REGISTER_COMPUTE_FN(Square);
    REGISTER_COMPUTE_FN(Tan);
    REGISTER_COMPUTE_FN(Tanh);

    // Additional compute functions not defined via UnaryOp functors.
    REGISTER_COMPUTE_FN(Elu);
    REGISTER_COMPUTE_FN(Relu);
    REGISTER_COMPUTE_FN(Relu6);
    REGISTER_COMPUTE_FN(Selu);
  }

  REGISTER_RELU_HELPER();

  // clang-format off
  REGISTER_COMPUTE_FN_HELPER(Abs,        functor::abs<T>);
  REGISTER_COMPUTE_FN_HELPER(Acos,       functor::acos<T>);
  REGISTER_COMPUTE_FN_HELPER(Acosh,      functor::acosh<T>);
  REGISTER_COMPUTE_FN_HELPER(Asin,       functor::asin<T>);
  REGISTER_COMPUTE_FN_HELPER(Asinh,      functor::asinh<T>);
  REGISTER_COMPUTE_FN_HELPER(Atan,       functor::atan<T>);
  REGISTER_COMPUTE_FN_HELPER(Atanh,      functor::atanh<T>);
  REGISTER_COMPUTE_FN_HELPER(Ceil,       functor::ceil<T>);
  REGISTER_COMPUTE_FN_HELPER(Cos,        functor::cos<T>);
  REGISTER_COMPUTE_FN_HELPER(Cosh,       functor::cosh<T>);
  REGISTER_COMPUTE_FN_HELPER(Expm1,      functor::expm1<T>);
  REGISTER_COMPUTE_FN_HELPER(Exp,        functor::exp<T>);
  REGISTER_COMPUTE_FN_HELPER(Floor,      functor::floor<T>);
  REGISTER_COMPUTE_FN_HELPER(Inv,        functor::inverse<T>);
  REGISTER_COMPUTE_FN_HELPER(Log,        functor::log<T>);
  REGISTER_COMPUTE_FN_HELPER(Log1p,      functor::log1p<T>);
  REGISTER_COMPUTE_FN_HELPER(Neg,        functor::neg<T>);
  REGISTER_COMPUTE_FN_HELPER(Reciprocal, functor::inverse<T>);
  REGISTER_COMPUTE_FN_HELPER(Rint,       functor::rint<T>);
  REGISTER_COMPUTE_FN_HELPER(Round,      functor::round<T>);
  REGISTER_COMPUTE_FN_HELPER(Rsqrt,      functor::rsqrt<T>);
  REGISTER_COMPUTE_FN_HELPER(Sigmoid,    functor::sigmoid<T>);
  REGISTER_COMPUTE_FN_HELPER(Sin,        functor::sin<T>);
  REGISTER_COMPUTE_FN_HELPER(Sinh,       functor::sinh<T>);
  REGISTER_COMPUTE_FN_HELPER(Sqrt,       functor::sqrt<T>);
  REGISTER_COMPUTE_FN_HELPER(Square,     functor::square<T>);
  REGISTER_COMPUTE_FN_HELPER(Tan,        functor::tan<T>);
  REGISTER_COMPUTE_FN_HELPER(Tanh,       functor::tanh<T>);
  // clang-format on
};

template <>
struct UnaryOpsCompositionSupport<Eigen::half>
    : UnaryOpsCompositionBase<Eigen::half> {
  using T = Eigen::half;

  UnaryOpsCompositionSupport() {
    REGISTER_COMPUTE_FN(Abs);
    REGISTER_COMPUTE_FN(Ceil);
    REGISTER_COMPUTE_FN(Cos);
    REGISTER_COMPUTE_FN(Expm1);
    REGISTER_COMPUTE_FN(Exp);
    REGISTER_COMPUTE_FN(Floor);
    REGISTER_COMPUTE_FN(Inv);
    REGISTER_COMPUTE_FN(Log);
    REGISTER_COMPUTE_FN(Log1p);
    REGISTER_COMPUTE_FN(Neg);
    REGISTER_COMPUTE_FN(Reciprocal);
    REGISTER_COMPUTE_FN(Round);
    REGISTER_COMPUTE_FN(Rsqrt);
    REGISTER_COMPUTE_FN(Sigmoid);
    REGISTER_COMPUTE_FN(Sin);
    REGISTER_COMPUTE_FN(Sqrt);
    REGISTER_COMPUTE_FN(Square);
    REGISTER_COMPUTE_FN(Tanh);
    // Additional compute functions not defined via UnaryOp functors.
    REGISTER_COMPUTE_FN(Elu);
    REGISTER_COMPUTE_FN(Relu);
    REGISTER_COMPUTE_FN(Relu6);
    REGISTER_COMPUTE_FN(Selu);
  }

  REGISTER_RELU_HELPER();

  // clang-format off
  REGISTER_COMPUTE_FN_HELPER(Abs,        functor::abs<T>);
  REGISTER_COMPUTE_FN_HELPER(Ceil,       functor::ceil<T>);
  REGISTER_COMPUTE_FN_HELPER(Cos,        functor::cos<T>);
  REGISTER_COMPUTE_FN_HELPER(Expm1,      functor::expm1<T>);
  REGISTER_COMPUTE_FN_HELPER(Exp,        functor::exp<T>);
  REGISTER_COMPUTE_FN_HELPER(Floor,      functor::floor<T>);
  REGISTER_COMPUTE_FN_HELPER(Inv,        functor::inverse<T>);
  REGISTER_COMPUTE_FN_HELPER(Log,        functor::log<T>);
  REGISTER_COMPUTE_FN_HELPER(Log1p,      functor::log1p<T>);
  REGISTER_COMPUTE_FN_HELPER(Neg,        functor::neg<T>);
  REGISTER_COMPUTE_FN_HELPER(Reciprocal, functor::inverse<T>);
  REGISTER_COMPUTE_FN_HELPER(Round,      functor::round<T>);
  REGISTER_COMPUTE_FN_HELPER(Rsqrt,      functor::rsqrt<T>);
  REGISTER_COMPUTE_FN_HELPER(Sigmoid,    functor::sigmoid<T>);
  REGISTER_COMPUTE_FN_HELPER(Sin,        functor::sin<T>);
  REGISTER_COMPUTE_FN_HELPER(Sqrt,       functor::sqrt<T>);
  REGISTER_COMPUTE_FN_HELPER(Square,     functor::square<T>);
  REGISTER_COMPUTE_FN_HELPER(Tanh,       functor::tanh<T>);
  // clang-format on
};

template <>
struct UnaryOpsCompositionSupport<double> : UnaryOpsCompositionBase<double> {
  using T = double;

  UnaryOpsCompositionSupport() {
    REGISTER_COMPUTE_FN(Abs);
    REGISTER_COMPUTE_FN(Acos);
    REGISTER_COMPUTE_FN(Acosh);
    REGISTER_COMPUTE_FN(Asin);
    REGISTER_COMPUTE_FN(Asinh);
    REGISTER_COMPUTE_FN(Atan);
    REGISTER_COMPUTE_FN(Atanh);
    REGISTER_COMPUTE_FN(Ceil);
    REGISTER_COMPUTE_FN(Cos);
    REGISTER_COMPUTE_FN(Cosh);
    REGISTER_COMPUTE_FN(Expm1);
    REGISTER_COMPUTE_FN(Exp);
    REGISTER_COMPUTE_FN(Floor);
    REGISTER_COMPUTE_FN(Inv);
    REGISTER_COMPUTE_FN(Log);
    REGISTER_COMPUTE_FN(Log1p);
    REGISTER_COMPUTE_FN(Neg);
    REGISTER_COMPUTE_FN(Reciprocal);
    REGISTER_COMPUTE_FN(Rint);
    REGISTER_COMPUTE_FN(Round);
    REGISTER_COMPUTE_FN(Rsqrt);
    REGISTER_COMPUTE_FN(Sigmoid);
    REGISTER_COMPUTE_FN(Sin);
    REGISTER_COMPUTE_FN(Sinh);
    REGISTER_COMPUTE_FN(Sqrt);
    REGISTER_COMPUTE_FN(Square);
    REGISTER_COMPUTE_FN(Tan);
    REGISTER_COMPUTE_FN(Tanh);
    // Additional compute functions not defined via UnaryOp functors.
    REGISTER_COMPUTE_FN(Elu);
    REGISTER_COMPUTE_FN(Relu);
    REGISTER_COMPUTE_FN(Relu6);
    REGISTER_COMPUTE_FN(Selu);
  }

  REGISTER_RELU_HELPER();

  // clang-format off
  REGISTER_COMPUTE_FN_HELPER(Abs,        functor::abs<T>);
  REGISTER_COMPUTE_FN_HELPER(Acos,       functor::acos<T>);
  REGISTER_COMPUTE_FN_HELPER(Acosh,      functor::acosh<T>);
  REGISTER_COMPUTE_FN_HELPER(Asin,       functor::asin<T>);
  REGISTER_COMPUTE_FN_HELPER(Asinh,      functor::asinh<T>);
  REGISTER_COMPUTE_FN_HELPER(Atan,       functor::atan<T>);
  REGISTER_COMPUTE_FN_HELPER(Atanh,      functor::atanh<T>);
  REGISTER_COMPUTE_FN_HELPER(Ceil,       functor::ceil<T>);
  REGISTER_COMPUTE_FN_HELPER(Cos,        functor::cos<T>);
  REGISTER_COMPUTE_FN_HELPER(Cosh,       functor::cosh<T>);
  REGISTER_COMPUTE_FN_HELPER(Expm1,      functor::expm1<T>);
  REGISTER_COMPUTE_FN_HELPER(Exp,        functor::exp<T>);
  REGISTER_COMPUTE_FN_HELPER(Floor,      functor::floor<T>);
  REGISTER_COMPUTE_FN_HELPER(Inv,        functor::inverse<T>);
  REGISTER_COMPUTE_FN_HELPER(Log,        functor::log<T>);
  REGISTER_COMPUTE_FN_HELPER(Log1p,      functor::log1p<T>);
  REGISTER_COMPUTE_FN_HELPER(Neg,        functor::neg<T>);
  REGISTER_COMPUTE_FN_HELPER(Reciprocal, functor::inverse<T>);
  REGISTER_COMPUTE_FN_HELPER(Rint,       functor::rint<T>);
  REGISTER_COMPUTE_FN_HELPER(Round,      functor::round<T>);
  REGISTER_COMPUTE_FN_HELPER(Rsqrt,      functor::rsqrt<T>);
  REGISTER_COMPUTE_FN_HELPER(Sigmoid,    functor::sigmoid<T>);
  REGISTER_COMPUTE_FN_HELPER(Sin,        functor::sin<T>);
  REGISTER_COMPUTE_FN_HELPER(Sinh,       functor::sinh<T>);
  REGISTER_COMPUTE_FN_HELPER(Sqrt,       functor::sqrt<T>);
  REGISTER_COMPUTE_FN_HELPER(Square,     functor::square<T>);
  REGISTER_COMPUTE_FN_HELPER(Tan,        functor::tan<T>);
  REGISTER_COMPUTE_FN_HELPER(Tanh,       functor::tanh<T>);
  // clang-format on
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_UnaryOpsComposition").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      UnaryOpsComposition<T>);

REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);
REGISTER_CPU(double);

#undef REGISTER_CPU

}  // namespace tensorflow
