// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cmath>

template <typename ValueType>
static ValueType batch_norm(ValueType elem, ValueType scale, ValueType offset,
                            ValueType mean, ValueType variance, float epsilon) {
  // offset_i + (scale_i * (X_i - mean_i) / (Sqrt[Var[X_i] + epsilon]))
  return offset + (scale * (elem - mean) /
                   static_cast<ValueType>(sqrtf(variance + epsilon)));
}

template <typename ValueType>
class BatchNormVertex : public poplar::Vertex {
 public:
  poplar::Input<poplar::Vector<ValueType>> operand;
  poplar::Input<ValueType> scale;
  poplar::Input<ValueType> offset;
  poplar::Input<ValueType> mean;
  poplar::Input<ValueType> variance;
  poplar::Input<float> epsilon;

  poplar::Output<poplar::Vector<ValueType>> result;

  bool compute() {
    for (std::size_t i = 0; i < operand.size(); ++i) {
      result[i] =
          batch_norm(operand[i], *scale, *offset, *mean, *variance, *epsilon);
    }

    return true;
  }
};

template class BatchNormVertex<float>;
template class BatchNormVertex<half>;
