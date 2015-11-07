// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_DEVICE_H
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_H

namespace Eigen {

/** \class TensorDevice
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Pseudo expression providing an operator = that will evaluate its argument
  * on the specified computing 'device' (GPU, thread pool, ...)
  *
  * Example:
  *    C.device(EIGEN_GPU) = A + B;
  *
  * Todo: thread pools.
  * Todo: operator +=, -=, *= and so on.
  */

template <typename ExpressionType, typename DeviceType> class TensorDevice {
  public:
    TensorDevice(const DeviceType& device, ExpressionType& expression) : m_device(device), m_expression(expression) {}

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE TensorDevice& operator=(const OtherDerived& other) {
      typedef TensorAssignOp<ExpressionType, const OtherDerived> Assign;
      Assign assign(m_expression, other);
      internal::TensorExecutor<const Assign, DeviceType>::run(assign, m_device);
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE TensorDevice& operator+=(const OtherDerived& other) {
      typedef typename OtherDerived::Scalar Scalar;
      typedef TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const ExpressionType, const OtherDerived> Sum;
      Sum sum(m_expression, other);
      typedef TensorAssignOp<ExpressionType, const Sum> Assign;
      Assign assign(m_expression, sum);
      internal::TensorExecutor<const Assign, DeviceType>::run(assign, m_device);
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE TensorDevice& operator-=(const OtherDerived& other) {
      typedef typename OtherDerived::Scalar Scalar;
      typedef TensorCwiseBinaryOp<internal::scalar_difference_op<Scalar>, const ExpressionType, const OtherDerived> Difference;
      Difference difference(m_expression, other);
      typedef TensorAssignOp<ExpressionType, const Difference> Assign;
      Assign assign(m_expression, difference);
      internal::TensorExecutor<const Assign, DeviceType>::run(assign, m_device);
      return *this;
    }

  protected:
    const DeviceType& m_device;
    ExpressionType& m_expression;
};


#ifdef EIGEN_USE_THREADS
template <typename ExpressionType> class TensorDevice<ExpressionType, ThreadPoolDevice> {
  public:
    TensorDevice(const ThreadPoolDevice& device, ExpressionType& expression) : m_device(device), m_expression(expression) {}

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE TensorDevice& operator=(const OtherDerived& other) {
      typedef TensorAssignOp<ExpressionType, const OtherDerived> Assign;
      Assign assign(m_expression, other);
      internal::TensorExecutor<const Assign, ThreadPoolDevice>::run(assign, m_device);
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE TensorDevice& operator+=(const OtherDerived& other) {
      typedef typename OtherDerived::Scalar Scalar;
      typedef TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const ExpressionType, const OtherDerived> Sum;
      Sum sum(m_expression, other);
      typedef TensorAssignOp<ExpressionType, const Sum> Assign;
      Assign assign(m_expression, sum);
      internal::TensorExecutor<const Assign, ThreadPoolDevice>::run(assign, m_device);
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE TensorDevice& operator-=(const OtherDerived& other) {
      typedef typename OtherDerived::Scalar Scalar;
      typedef TensorCwiseBinaryOp<internal::scalar_difference_op<Scalar>, const ExpressionType, const OtherDerived> Difference;
      Difference difference(m_expression, other);
      typedef TensorAssignOp<ExpressionType, const Difference> Assign;
      Assign assign(m_expression, difference);
      internal::TensorExecutor<const Assign, ThreadPoolDevice>::run(assign, m_device);
      return *this;
    }

  protected:
    const ThreadPoolDevice& m_device;
    ExpressionType& m_expression;
};
#endif

#if defined(EIGEN_USE_GPU)
template <typename ExpressionType> class TensorDevice<ExpressionType, GpuDevice>
{
  public:
    TensorDevice(const GpuDevice& device, ExpressionType& expression) : m_device(device), m_expression(expression) {}

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE TensorDevice& operator=(const OtherDerived& other) {
      typedef TensorAssignOp<ExpressionType, const OtherDerived> Assign;
      Assign assign(m_expression, other);
      internal::TensorExecutor<const Assign, GpuDevice>::run(assign, m_device);
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE TensorDevice& operator+=(const OtherDerived& other) {
      typedef typename OtherDerived::Scalar Scalar;
      typedef TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const ExpressionType, const OtherDerived> Sum;
      Sum sum(m_expression, other);
      typedef TensorAssignOp<ExpressionType, const Sum> Assign;
      Assign assign(m_expression, sum);
      internal::TensorExecutor<const Assign, GpuDevice>::run(assign, m_device);
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE TensorDevice& operator-=(const OtherDerived& other) {
      typedef typename OtherDerived::Scalar Scalar;
      typedef TensorCwiseBinaryOp<internal::scalar_difference_op<Scalar>, const ExpressionType, const OtherDerived> Difference;
      Difference difference(m_expression, other);
      typedef TensorAssignOp<ExpressionType, const Difference> Assign;
      Assign assign(m_expression, difference);
      internal::TensorExecutor<const Assign, GpuDevice>::run(assign, m_device);
      return *this;
    }

  protected:
    const GpuDevice& m_device;
    ExpressionType& m_expression;
};
#endif


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_H
