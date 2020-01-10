#ifndef TENSORFLOW_KERNELS_REVERSE_SEQUENCE_OP_H_
#define TENSORFLOW_KERNELS_REVERSE_SEQUENCE_OP_H_
// Generator definition for ReverseSequenceOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

namespace generator {

template <typename T, size_t Dims>
class ReverseGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ReverseGenerator(typename TTypes<T, Dims>::ConstTensor input, int32 seq_dim,
                   TTypes<int64>::ConstVec seq_lengths)
      : input_(input), seq_dim_(seq_dim), seq_lengths_(seq_lengths) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, Dims>& coords) const {
    Eigen::array<Eigen::DenseIndex, Dims> new_coords = coords;
    if (coords[seq_dim_] < seq_lengths_(coords[0])) {
      new_coords[seq_dim_] = seq_lengths_(coords[0]) - coords[seq_dim_] - 1;
    }

    return input_(new_coords);
  }

 private:
  typename TTypes<T, Dims>::ConstTensor input_;
  int32 seq_dim_;
  TTypes<int64>::ConstVec seq_lengths_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T, size_t Dims>
struct ReverseSequence {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,
      int32 seq_dim, TTypes<int64>::ConstVec seq_lengths,
      typename TTypes<T, Dims>::Tensor output) {
    generator::ReverseGenerator<T, Dims> generator(input, seq_dim, seq_lengths);
    output.device(d) = input.generate(generator);
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_REVERSE_SEQUENCE_OP_H_
