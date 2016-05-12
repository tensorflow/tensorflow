#ifndef TENSORFLOW_KERNELS_ATTENTION_MASK_OPS_H_
#define TENSORFLOW_KERNELS_ATTENTION_MASK_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace generator {

class AttentionMaskGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  AttentionMaskGenerator(
      float fill_value, TTypes<int64>::ConstVec sequence_len,
      TTypes<float>::ConstMatrix input)
    : fill_value_(fill_value), sequence_len_(sequence_len), input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  float operator()(const Eigen::array<Eigen::DenseIndex, 2>& coords) const {
    if (coords[1] < sequence_len_(coords[0])) {
      return input_(coords);
    } else {
      return fill_value_;
    }
  }

 private:
  float fill_value_;
  TTypes<int64>::ConstVec sequence_len_;
  TTypes<float>::ConstMatrix input_;
};

class AttentionMaskMedianGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  AttentionMaskMedianGenerator(
      float fill_value, int window_l, int window_r,
      TTypes<int64>::ConstVec sequence_len, TTypes<int64>::ConstVec median,
      TTypes<float>::ConstMatrix input)
    : fill_value_(fill_value), window_l_(window_l), window_r_(window_r),
      sequence_len_(sequence_len), median_(median), input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  float operator()(const Eigen::array<Eigen::DenseIndex, 2>& coords) const {
    const int median = median_(coords[0]);
    const int idx_min = median - window_l_;
    const int idx_max = median + window_r_;
    const int idx = coords[1];
    if (idx >= idx_min && idx <= idx_max && idx < sequence_len_(coords[0])) {
      return input_(coords);
    } else {
      return fill_value_;
    }
  }

 private:
  float fill_value_;
  int window_l_;
  int window_r_;
  TTypes<int64>::ConstVec sequence_len_;
  TTypes<int64>::ConstVec median_;
  TTypes<float>::ConstMatrix input_;
};

}  // end namespace generator

namespace functor {

template <typename Device>
struct AttentionMask {
  EIGEN_ALWAYS_INLINE static
  void Compute(
      const Device& d, float fill_value,
      typename TTypes<int64>::ConstVec sequence_len,
      typename TTypes<float>::ConstMatrix input,
      typename TTypes<float>::Matrix output) {
    generator::AttentionMaskGenerator generator(
        fill_value, sequence_len, input);
    output.device(d) = input.generate(generator);
  }
};

template <typename Device>
struct AttentionMaskMedian {
  EIGEN_ALWAYS_INLINE static
  void Compute(
      const Device& d, float fill_value, int64 window_l, int64 window_r,
      typename TTypes<int64>::ConstVec sequence_len,
      typename TTypes<float>::ConstMatrix input,
      typename TTypes<int64>::ConstVec median,
      typename TTypes<float>::Matrix output) {
    generator::AttentionMaskMedianGenerator generator(
        fill_value, window_l, window_r, sequence_len, median, input);
    output.device(d) = input.generate(generator);
  }
};

template <typename Device>
struct ComputeMedian {
  EIGEN_ALWAYS_INLINE static
  void Compute(
      const Device& d, typename TTypes<float>::ConstMatrix input,
      typename TTypes<int64>::Vec median);
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_ATTENTION_MASK_OPS_H_
