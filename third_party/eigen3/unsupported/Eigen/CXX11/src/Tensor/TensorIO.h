// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_IO_H
#define EIGEN_CXX11_TENSOR_TENSOR_IO_H

namespace Eigen {

namespace internal {
template<>
struct significant_decimals_impl<std::string>
    : significant_decimals_default_impl<std::string, true>
{};
}


template <typename T>
std::ostream& operator << (std::ostream& os, const TensorBase<T, ReadOnlyAccessors>& expr) {
  // Evaluate the expression if needed
  TensorForcedEvalOp<const T> eval = expr.eval();
  TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice> tensor(eval, DefaultDevice());
  tensor.evalSubExprsIfNeeded(NULL);

  typedef typename internal::remove_const<typename T::Scalar>::type Scalar;
  typedef typename T::Index Index;
  typedef typename TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice>::Dimensions Dimensions;
  const Index total_size = internal::array_prod(tensor.dimensions());

  // Print the tensor as a 1d vector or a 2d matrix.
  static const int rank = internal::array_size<Dimensions>::value;
  if (rank == 0) {
    os << tensor.coeff(0);
  } else if (rank == 1) {
    Map<const Array<Scalar, Dynamic, 1> > array(const_cast<Scalar*>(tensor.data()), total_size);
    os << array;
  } else {
    const Index first_dim = tensor.dimensions()[0];
    static const int layout = TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice>::Layout;
    Map<const Array<Scalar, Dynamic, Dynamic, layout> > matrix(const_cast<Scalar*>(tensor.data()), first_dim, total_size/first_dim);
    os << matrix;
  }

  // Cleanup.
  tensor.cleanup();
  return os;
}

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_IO_H
