// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_BASE_H
#define EIGEN_CXX11_TENSOR_TENSOR_BASE_H

// clang-format off

namespace Eigen {

/** \class TensorBase
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor base class.
  *
  * This class is the common parent of the Tensor and TensorMap class, thus
  * making it possible to use either class interchangably in expressions.
  */

template<typename Derived>
class TensorBase<Derived, ReadOnlyAccessors>
{
  public:
    typedef internal::traits<Derived> DerivedTraits;
    typedef typename DerivedTraits::Scalar Scalar;
    typedef typename DerivedTraits::Index Index;
    typedef typename internal::remove_const<Scalar>::type CoeffReturnType;
    typedef typename internal::packet_traits<CoeffReturnType>::type PacketReturnType;
    static const int NumDimensions = DerivedTraits::NumDimensions;

    // Generic nullary operation support.
    template <typename CustomNullaryOp> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<CustomNullaryOp, const Derived>
    nullaryExpr(const CustomNullaryOp& func) const {
      return TensorCwiseNullaryOp<CustomNullaryOp, const Derived>(derived(), func);
    }

    // Coefficient-wise nullary operators
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived>
    constant(const Scalar& value) const {
      return nullaryExpr(internal::scalar_constant_op<Scalar>(value));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<internal::UniformRandomGenerator<Scalar>, const Derived>
    random() const {
      return nullaryExpr(internal::UniformRandomGenerator<Scalar>());
    }
    template <typename RandomGenerator> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<RandomGenerator, const Derived>
    random(const RandomGenerator& gen = RandomGenerator()) const {
      return nullaryExpr(gen);
    }

    // Tensor generation
    template <typename Generator> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorGeneratorOp<Generator, const Derived>
    generate(const Generator& generator) const {
      return TensorGeneratorOp<Generator, const Derived>(derived(), generator);
    }

    // Generic unary operation support.
    template <typename CustomUnaryOp> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<CustomUnaryOp, const Derived>
    unaryExpr(const CustomUnaryOp& func) const {
      return TensorCwiseUnaryOp<CustomUnaryOp, const Derived>(derived(), func);
    }

    // Coefficient-wise unary operators
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_opposite_op<Scalar>, const Derived>
    operator-() const {
      return unaryExpr(internal::scalar_opposite_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_sqrt_op<Scalar>, const Derived>
    sqrt() const {
      return unaryExpr(internal::scalar_sqrt_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_rsqrt_op<Scalar>, const Derived>
    rsqrt() const {
      return unaryExpr(internal::scalar_rsqrt_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_square_op<Scalar>, const Derived>
    square() const {
      return unaryExpr(internal::scalar_square_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_cube_op<Scalar>, const Derived>
    cube() const {
      return unaryExpr(internal::scalar_cube_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const Derived>
    inverse() const {
      return unaryExpr(internal::scalar_inverse_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_tanh_op<Scalar>, const Derived>
    tanh() const {
      return unaryExpr(internal::scalar_tanh_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_sigmoid_op<Scalar>, const Derived>
    sigmoid() const {
      return unaryExpr(internal::scalar_sigmoid_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_exp_op<Scalar>, const Derived>
    exp() const {
      return unaryExpr(internal::scalar_exp_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_log_op<Scalar>, const Derived>
    log() const {
      return unaryExpr(internal::scalar_log_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived>
    abs() const {
      return unaryExpr(internal::scalar_abs_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_pow_op<Scalar>, const Derived>
    pow(Scalar exponent) const {
      return unaryExpr(internal::scalar_pow_op<Scalar>(exponent));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_add_op<Scalar>, const Derived>
    operator+ (Scalar rhs) const {
      return unaryExpr(internal::scalar_add_op<Scalar>(rhs));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_sub_op<Scalar>, const Derived>
    operator- (Scalar rhs) const {
      EIGEN_STATIC_ASSERT((std::numeric_limits<Scalar>::is_signed || internal::is_same<Scalar, const std::complex<float> >::value), YOU_MADE_A_PROGRAMMING_MISTAKE);
      return unaryExpr(internal::scalar_sub_op<Scalar>(rhs));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_multiple_op<Scalar>, const Derived>
    operator* (Scalar rhs) const {
      return unaryExpr(internal::scalar_multiple_op<Scalar>(rhs));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_quotient1_op<Scalar>, const Derived>
    operator/ (Scalar rhs) const {
      // EIGEN_STATIC_ASSERT(!std::numeric_limits<Scalar>::is_integer, YOU_MADE_A_PROGRAMMING_MISTAKE);
      return unaryExpr(internal::scalar_quotient1_op<Scalar>(rhs));
    }

    template <typename Scale>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_multiple2_op<Scalar, Scale>, const Derived>
    scale (Scale rhs) const {
      return unaryExpr(internal::scalar_multiple2_op<Scalar, Scale>(rhs));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_mod_op<Scalar>, const Derived>
    operator% (Scalar rhs) const {
      EIGEN_STATIC_ASSERT(std::numeric_limits<Scalar>::is_integer, YOU_MADE_A_PROGRAMMING_MISTAKE_TRY_MOD);
      return unaryExpr(internal::scalar_mod_op<Scalar>(rhs));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<internal::scalar_fmod_op<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    mod(Scalar rhs) const {
      EIGEN_STATIC_ASSERT(!std::numeric_limits<Scalar>::is_integer, YOU_MADE_A_PROGRAMMING_MISTAKE_FMOD_IS_NOT_FOR_INTEGERS);
      return mod(constant(rhs));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<internal::scalar_max_op<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    cwiseMax(Scalar threshold) const {
      return cwiseMax(constant(threshold));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<internal::scalar_min_op<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    cwiseMin(Scalar threshold) const {
      return cwiseMin(constant(threshold));
    }

    template <typename NewType> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorConversionOp<NewType, const Derived>
    cast() const {
      return TensorConversionOp<NewType, const Derived>(derived());
    }

    // Generic binary operation support.
    template <typename CustomBinaryOp, typename OtherDerived> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<CustomBinaryOp, const Derived, const OtherDerived>
    binaryExpr(const OtherDerived& other, const CustomBinaryOp& func) const {
      return TensorCwiseBinaryOp<CustomBinaryOp, const Derived, const OtherDerived>(derived(), other, func);
    }

    // Coefficient-wise binary operators.
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const Derived, const OtherDerived>
    operator+(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_sum_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_difference_op<Scalar>, const Derived, const OtherDerived>
    operator-(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_difference_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_product_op<Scalar>, const Derived, const OtherDerived>
    operator*(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_product_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_quotient_op<Scalar>, const Derived, const OtherDerived>
    operator/(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_quotient_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_fmod_op<Scalar>, const Derived, const OtherDerived>
    mod(const OtherDerived& other) const {
      EIGEN_STATIC_ASSERT(!std::numeric_limits<Scalar>::is_integer, YOU_MADE_A_PROGRAMMING_MISTAKE_FMOD_IS_NOT_FOR_INTEGERS);
      return binaryExpr(other.derived(), internal::scalar_fmod_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_max_op<Scalar>, const Derived, const OtherDerived>
    cwiseMax(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_max_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_min_op<Scalar>, const Derived, const OtherDerived>
    cwiseMin(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_min_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_boolean_and_op, const Derived, const OtherDerived>
    operator&&(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_boolean_and_op());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_boolean_or_op, const Derived, const OtherDerived>
    operator||(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_boolean_or_op());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_boolean_xor_op, const Derived, const OtherDerived>
    operator^(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_boolean_xor_op());
    }

    // Comparisons and tests.
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::less<Scalar>, const Derived, const OtherDerived>
    operator<(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::less<Scalar>());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::less_equal<Scalar>, const Derived, const OtherDerived>
    operator<=(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::less_equal<Scalar>());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::greater<Scalar>, const Derived, const OtherDerived>
    operator>(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::greater<Scalar>());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::greater_equal<Scalar>, const Derived, const OtherDerived>
    operator>=(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::greater_equal<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::equal_to<Scalar>, const Derived, const OtherDerived>
    operator==(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::equal_to<Scalar>());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::not_equal_to<Scalar>, const Derived, const OtherDerived>
    operator!=(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::not_equal_to<Scalar>());
    }

    // comparisons and tests for Scalars
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<std::less<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    operator<(Scalar threshold) const {
      return operator<(constant(threshold));
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<std::less_equal<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    operator<=(Scalar threshold) const {
      return operator<=(constant(threshold));
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<std::greater<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    operator>(Scalar threshold) const {
      return operator>(constant(threshold));
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<std::greater_equal<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    operator>=(Scalar threshold) const {
      return operator>=(constant(threshold));
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<std::equal_to<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    operator==(Scalar threshold) const {
      return operator==(constant(threshold));
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<std::not_equal_to<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    operator!=(Scalar threshold) const {
      return operator!=(constant(threshold));
    }

    // Coefficient-wise ternary operators.
    template<typename ThenDerived, typename ElseDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorSelectOp<const Derived, const ThenDerived, const ElseDerived>
    select(const ThenDerived& thenTensor, const ElseDerived& elseTensor) const {
      return TensorSelectOp<const Derived, const ThenDerived, const ElseDerived>(derived(), thenTensor.derived(), elseTensor.derived());
    }

    // Contractions.
    typedef Eigen::IndexPair<Index> DimensionPair;

    template<typename OtherDerived, typename Dimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorContractionOp<const Dimensions, const Derived, const OtherDerived>
    contract(const OtherDerived& other, const Dimensions& dims) const {
      return TensorContractionOp<const Dimensions, const Derived, const OtherDerived>(derived(), other.derived(), dims);
    }

    // Convolutions.
    template<typename KernelDerived, typename Dimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorConvolutionOp<const Dimensions, const Derived, const KernelDerived>
    convolve(const KernelDerived& kernel, const Dimensions& dims) const {
      return TensorConvolutionOp<const Dimensions, const Derived, const KernelDerived>(derived(), kernel.derived(), dims);
    }

    // Convolutions by fft.
    template<typename KernelDerived, typename Dimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorConvolutionByFFTOp<const Dimensions, const Derived, const KernelDerived>
    convolvebyfft(const KernelDerived& kernel, const Dimensions& dims) const {
      return TensorConvolutionByFFTOp<const Dimensions, const Derived, const KernelDerived>(derived(), kernel.derived(), dims);
    }

    // Reductions.
    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::SumReducer<CoeffReturnType>, const Dims, const Derived>
    sum(const Dims& dims) const {
      return TensorReductionOp<internal::SumReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::SumReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::SumReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>
    sum() const {
      DimensionList<Index, NumDimensions> in_dims;
      return TensorReductionOp<internal::SumReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::SumReducer<CoeffReturnType>());
    }

    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::MeanReducer<CoeffReturnType>, const Dims, const Derived>
    mean(const Dims& dims) const {
      return TensorReductionOp<internal::MeanReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::MeanReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::MeanReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>
    mean() const {
      DimensionList<Index, NumDimensions> in_dims;
      return TensorReductionOp<internal::MeanReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::MeanReducer<CoeffReturnType>());
    }

    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::ProdReducer<CoeffReturnType>, const Dims, const Derived>
    prod(const Dims& dims) const {
      return TensorReductionOp<internal::ProdReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::ProdReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::ProdReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>
    prod() const {
      DimensionList<Index, NumDimensions> in_dims;
      return TensorReductionOp<internal::ProdReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::ProdReducer<CoeffReturnType>());
    }

    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::MaxReducer<CoeffReturnType>, const Dims, const Derived>
    maximum(const Dims& dims) const {
      return TensorReductionOp<internal::MaxReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::MaxReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::MaxReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>
    maximum() const {
      DimensionList<Index, NumDimensions> in_dims;
      return TensorReductionOp<internal::MaxReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::MaxReducer<CoeffReturnType>());
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorTupleReducerOp<
      internal::ArgMaxTupleReducer<Tuple<Index, CoeffReturnType> >,
      const array<Index, NumDimensions>, const Derived>
    argmax() const {
      array<Index, NumDimensions> in_dims;
      for (int d = 0; d < NumDimensions; ++d) in_dims[d] = d;
      return TensorTupleReducerOp<
        internal::ArgMaxTupleReducer<Tuple<Index, CoeffReturnType> >,
        const array<Index, NumDimensions>,
        const Derived>(derived(), internal::ArgMaxTupleReducer<Tuple<Index, CoeffReturnType> >(), -1, in_dims);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorTupleReducerOp<
      internal::ArgMinTupleReducer<Tuple<Index, CoeffReturnType> >,
      const array<Index, NumDimensions>, const Derived>
    argmin() const {
      array<Index, NumDimensions> in_dims;
      for (int d = 0; d < NumDimensions; ++d) in_dims[d] = d;
      return TensorTupleReducerOp<
        internal::ArgMinTupleReducer<Tuple<Index, CoeffReturnType> >,
        const array<Index, NumDimensions>,
        const Derived>(derived(), internal::ArgMinTupleReducer<Tuple<Index, CoeffReturnType> >(), -1, in_dims);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorTupleReducerOp<
      internal::ArgMaxTupleReducer<Tuple<Index, CoeffReturnType> >,
      const array<Index, 1>, const Derived>
    argmax(const int return_dim) const {
      array<Index, 1> in_dims;
      in_dims[0] = return_dim;
      return TensorTupleReducerOp<
        internal::ArgMaxTupleReducer<Tuple<Index, CoeffReturnType> >,
        const array<Index, 1>,
        const Derived>(derived(), internal::ArgMaxTupleReducer<Tuple<Index, CoeffReturnType> >(), return_dim, in_dims);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorTupleReducerOp<
      internal::ArgMinTupleReducer<Tuple<Index, CoeffReturnType> >,
      const array<Index, 1>, const Derived>
    argmin(const int return_dim) const {
      array<Index, 1> in_dims;
      in_dims[0] = return_dim;
      return TensorTupleReducerOp<
        internal::ArgMinTupleReducer<Tuple<Index, CoeffReturnType> >,
        const array<Index, 1>,
        const Derived>(derived(), internal::ArgMinTupleReducer<Tuple<Index, CoeffReturnType> >(), return_dim, in_dims);
    }

    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::MinReducer<CoeffReturnType>, const Dims, const Derived>
    minimum(const Dims& dims) const {
      return TensorReductionOp<internal::MinReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::MinReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::MinReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>
    minimum() const {
      DimensionList<Index, NumDimensions> in_dims;
      return TensorReductionOp<internal::MinReducer<CoeffReturnType>, const DimensionList<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::MinReducer<CoeffReturnType>());
    }

    // This does not short-circuit, so is potentially very inefficient.
    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::AndReducer, const Dims, const TensorConversionOp<bool, const Derived> >
    all(const Dims& dims) const {
      return cast<bool>().reduce(dims, internal::AndReducer());
    }

    // This does not short-circuit, so is potentially very inefficient.
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::AndReducer, const DimensionList<Index, NumDimensions>, const TensorConversionOp<bool, const Derived> >
    all() const {
      DimensionList<Index, NumDimensions> in_dims;
      return cast<bool>().reduce(in_dims, internal::AndReducer());
    }

    // This does not short-circuit, so is potentially very inefficient.
    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::OrReducer, const Dims, const TensorConversionOp<bool, const Derived> >
    any(const Dims& dims) const {
      return cast<bool>().reduce(dims, internal::OrReducer());
    }

    // This does not short-circuit, so is potentially very inefficient.
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::OrReducer, const DimensionList<Index, NumDimensions>, const TensorConversionOp<bool, const Derived> >
    any() const {
      DimensionList<Index, NumDimensions> in_dims;
      return cast<bool>().reduce(in_dims, internal::OrReducer());
    }

    template <typename Reducer, typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<Reducer, const Dims, const Derived>
    reduce(const Dims& dims, const Reducer& reducer) const {
      return TensorReductionOp<Reducer, const Dims, const Derived>(derived(), dims, reducer);
    }

    template <typename Broadcast> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorBroadcastingOp<const Broadcast, const Derived>
    broadcast(const Broadcast& broadcast) const {
      return TensorBroadcastingOp<const Broadcast, const Derived>(derived(), broadcast);
    }

    template <int FFTDataType, int FFTDirection, typename FFT> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorFFTOp<const FFT, const Derived, FFTDataType, FFTDirection>
    fft(const FFT& fft) const {
      return TensorFFTOp<const FFT, const Derived, FFTDataType, FFTDirection>(derived(), fft);
    }

    template <typename Axis, typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorConcatenationOp<Axis, const Derived, const OtherDerived>
    concatenate(const OtherDerived& other, Axis axis) const {
      return TensorConcatenationOp<Axis, const Derived, const OtherDerived>(derived(), other.derived(), axis);
    }

    template <typename PatchDims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorPatchOp<const PatchDims, const Derived>
    extract_patches(const PatchDims& patch_dims) const {
      return TensorPatchOp<const PatchDims, const Derived>(derived(), patch_dims);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic, const Derived>
    extract_volume_patches(const Index patch_planes, const Index patch_rows, const Index patch_cols,
                           const Index plane_stride = 1, const Index row_stride = 1, const Index col_stride = 1,
                           const PaddingType padding_type = PADDING_SAME, const Scalar padding_value = 0) const {
      return TensorVolumePatchOp<Dynamic, Dynamic, Dynamic, const Derived>(derived(), patch_planes, patch_rows, patch_cols, plane_stride, row_stride, col_stride, 1, 1, 1, 1, 1, 1, padding_type, padding_value);
    }


    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic, const Derived>
    extract_volume_patches(const Index patch_planes, const Index patch_rows, const Index patch_cols,
                           const Index plane_stride, const Index row_stride, const Index col_stride,
                           const Index plane_inflate_stride, const Index row_inflate_stride, const Index col_inflate_stride,
                           const Index padding_top_z, const Index padding_bottom_z,
                           const Index padding_top, const Index padding_bottom,
                           const Index padding_left, const Index padding_right, const Scalar padding_value = 0) const {
      return TensorVolumePatchOp<Dynamic, Dynamic, Dynamic, const Derived>(derived(), patch_planes, patch_rows, patch_cols, plane_stride, row_stride, col_stride, 1, 1, 1, plane_inflate_stride, row_inflate_stride, col_inflate_stride, padding_top_z, padding_bottom_z, padding_top, padding_bottom, padding_left, padding_right, padding_value);
    }

    template <Index Rows, Index Cols> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Rows, Cols, const Derived>
    extract_image_patches() const {
      return TensorImagePatchOp<Rows, Cols, const Derived>(derived(), Rows, Cols, 1, 1, 1, 1, 1, 1, PADDING_SAME, 0);
    }

    template <Index Rows, Index Cols> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Rows, Cols, const Derived>
    extract_image_patches(const PaddingType padding_type) const {
      return TensorImagePatchOp<Rows, Cols, const Derived>(derived(), Rows, Cols, 1, 1, 1, 1, 1, 1, padding_type, 0);
    }

    template <Index Rows, Index Cols> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Rows, Cols, const Derived>
    extract_image_patches(const Index stride, const PaddingType padding_type) const {
      return TensorImagePatchOp<Rows, Cols, const Derived>(derived(), Rows, Cols, stride, stride, 1, 1, 1, 1, padding_type, 0);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride = 1, const Index col_stride = 1) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 1, 1, 1, 1, PADDING_SAME, 0);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride, const Index col_stride,
                          const PaddingType padding_type) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 1, 1, 1, 1, padding_type, 0);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride, const Index col_stride,
                          const PaddingType padding_type, const Scalar padding_value) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 1, 1, 1, 1, padding_type, padding_value);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride, const Index col_stride,
                          const Index in_row_stride, const Index in_col_stride) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 in_row_stride, in_col_stride, 1, 1, PADDING_SAME, 0);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride, const Index col_stride,
                          const Index in_row_stride, const Index in_col_stride,
                          const PaddingType padding_type) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 in_row_stride, in_col_stride, 1, 1, padding_type, 0);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride, const Index col_stride,
                          const Index in_row_stride, const Index in_col_stride,
                          const PaddingType padding_type, const Scalar padding_value) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 in_row_stride, in_col_stride, 1, 1, padding_type, padding_value);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride, const Index col_stride,
                          const Index in_row_stride, const Index in_col_stride,
                          const Index row_inflate_stride, const Index col_inflate_stride,
                          const PaddingType padding_type, const Scalar padding_value) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 in_row_stride, in_col_stride, row_inflate_stride, col_inflate_stride,
                                                                 padding_type, padding_value);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride, const Index col_stride,
                          const Index in_row_stride, const Index in_col_stride,
                          const Index row_inflate_stride, const Index col_inflate_stride,
                          const Index padding_top, const Index padding_bottom,
                          const Index padding_left,const Index padding_right,
                          const Scalar padding_value) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 in_row_stride, in_col_stride, row_inflate_stride, col_inflate_stride,
                                                                 padding_top, padding_bottom, padding_left, padding_right, padding_value);
    }

    // Morphing operators.
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorLayoutSwapOp<const Derived>
    swap_layout() const {
      return TensorLayoutSwapOp<const Derived>(derived());
    }
    template <typename NewDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReshapingOp<const NewDimensions, const Derived>
    reshape(const NewDimensions& newDimensions) const {
      return TensorReshapingOp<const NewDimensions, const Derived>(derived(), newDimensions);
    }
    template <typename StartIndices, typename Sizes> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorSlicingOp<const StartIndices, const Sizes, const Derived>
    slice(const StartIndices& startIndices, const Sizes& sizes) const {
      return TensorSlicingOp<const StartIndices, const Sizes, const Derived>(derived(), startIndices, sizes);
    }
    template <Index DimId> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorChippingOp<DimId, const Derived>
    chip(const Index offset) const {
      return TensorChippingOp<DimId, const Derived>(derived(), offset, DimId);
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorChippingOp<Dynamic, const Derived>
    chip(const Index offset, const Index dim) const {
      return TensorChippingOp<Dynamic, const Derived>(derived(), offset, dim);
    }
    template <typename ReverseDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReverseOp<const ReverseDimensions, const Derived>
    reverse(const ReverseDimensions& rev) const {
      return TensorReverseOp<const ReverseDimensions, const Derived>(derived(), rev);
    }
    template <typename PaddingDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorPaddingOp<const PaddingDimensions, const Derived>
    pad(const PaddingDimensions& padding) const {
      return TensorPaddingOp<const PaddingDimensions, const Derived>(derived(), padding, Scalar(0));
    }
    template <typename PaddingDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorPaddingOp<const PaddingDimensions, const Derived>
    pad (const PaddingDimensions& padding, const Scalar padding_value) const {
      return TensorPaddingOp<const PaddingDimensions, const Derived>(derived(), padding, padding_value);
    }
    template <typename Shuffle> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorShufflingOp<const Shuffle, const Derived>
    shuffle(const Shuffle& shuffle) const {
      return TensorShufflingOp<const Shuffle, const Derived>(derived(), shuffle);
    }
    template <typename Strides> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorStridingOp<const Strides, const Derived>
    stride(const Strides& strides) const {
      return TensorStridingOp<const Strides, const Derived>(derived(), strides);
    }
    template <typename Strides> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorInflationOp<const Strides, const Derived>
    inflate(const Strides& strides) const {
      return TensorInflationOp<const Strides, const Derived>(derived(), strides);
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorTrueIndicesOp<const Derived>
    true_indices(const Index& not_true_value = -1) const {
      return TensorTrueIndicesOp<const Derived>(derived(), not_true_value);
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorIndexTupleOp<const Derived>
    index_tuples() const {
      return TensorIndexTupleOp<const Derived>(derived());
    }
    template <typename CustomUnaryFunc>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCustomUnaryOp<const CustomUnaryFunc, const Derived> customOp(const CustomUnaryFunc& op) const {
      return TensorCustomUnaryOp<const CustomUnaryFunc, const Derived>(derived(), op);
    }
    template <typename OtherDerived, typename CustomBinaryFunc>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCustomBinaryOp<const CustomBinaryFunc, const Derived, const OtherDerived> customOp(const OtherDerived& other, const CustomBinaryFunc& op) const {
      return TensorCustomBinaryOp<const CustomBinaryFunc, const Derived, const OtherDerived>(derived(), other, op);
    }

    // Force the evaluation of the expression.
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorForcedEvalOp<const Derived> eval() const {
      return TensorForcedEvalOp<const Derived>(derived());
    }

  protected:
    template <typename Scalar, std::size_t NumIndices, int Options, typename IndexType> friend class Tensor;
    template <typename Scalar, int Option, typename IndexTypes> friend class TensorVarDim;
    template <typename Scalar, typename Dimensions, int Option, typename IndexTypes> friend class TensorFixedSize;
    template <typename OtherDerived, int AccessLevel> friend class TensorBase;
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

template<typename Derived>
class TensorBase<Derived, WriteAccessors> : public TensorBase<Derived, ReadOnlyAccessors> {
 public:
    typedef internal::traits<Derived> DerivedTraits;
    typedef typename DerivedTraits::Scalar Scalar;
    typedef typename DerivedTraits::Index Index;
    typedef Scalar CoeffReturnType;
    typedef typename internal::packet_traits<Scalar>::type PacketReturnType;
    static const int NumDimensions = DerivedTraits::NumDimensions;

    template <typename Scalar, std::size_t NumIndices, int Options, typename IndexType> friend class Tensor;
    template <typename Scalar, int Options, typename IndexType> friend class TensorVarDim;
    template <typename OtherDerived, int AccessLevel> friend class TensorBase;

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setZero() {
      return setConstant(Scalar(0));
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setConstant(const Scalar& val) {
      return derived() = this->constant(val);
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setRandom() {
      return derived() = this->random();
    }
    template <typename RandomGenerator> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setRandom() {
      return derived() = this->template random<RandomGenerator>();
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setValues(
        const typename internal::Initializer<Derived, NumDimensions>::InitList& vals) {
      TensorEvaluator<Derived, DefaultDevice> eval(derived(), DefaultDevice());
      internal::initialize_tensor<Derived, NumDimensions>(eval, vals);
      return derived();
    }
#endif  // EIGEN_HAS_VARIADIC_TEMPLATES

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator+=(const OtherDerived& other) {
      return derived() = derived() + other.derived();
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator-=(const OtherDerived& other) {
      return derived() = derived() - other.derived();
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator*=(const OtherDerived& other) {
      return derived() = derived() * other.derived();
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator/=(const OtherDerived& other) {
      return derived() = derived() / other.derived();
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorLayoutSwapOp<const Derived>
    swap_layout() const {
      return TensorLayoutSwapOp<const Derived>(derived());
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorLayoutSwapOp<Derived>
    swap_layout() {
      return TensorLayoutSwapOp<Derived>(derived());
    }

    template <typename Axis, typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorConcatenationOp<const Axis, const Derived, const OtherDerived>
    concatenate(const OtherDerived& other, const Axis& axis) const {
      return TensorConcatenationOp<const Axis, const Derived, const OtherDerived>(derived(), other, axis);
    }
    template <typename Axis, typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorConcatenationOp<const Axis, Derived, OtherDerived>
    concatenate(const OtherDerived& other, const Axis& axis) {
      return TensorConcatenationOp<const Axis, Derived, OtherDerived>(derived(), other, axis);
    }

    template <typename NewDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReshapingOp<const NewDimensions, const Derived>
    reshape(const NewDimensions& newDimensions) const {
      return TensorReshapingOp<const NewDimensions, const Derived>(derived(), newDimensions);
    }
    template <typename NewDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorReshapingOp<const NewDimensions, Derived>
    reshape(const NewDimensions& newDimensions) {
      return TensorReshapingOp<const NewDimensions, Derived>(derived(), newDimensions);
    }

    template <typename StartIndices, typename Sizes> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorSlicingOp<const StartIndices, const Sizes, const Derived>
    slice(const StartIndices& startIndices, const Sizes& sizes) const {
      return TensorSlicingOp<const StartIndices, const Sizes, const Derived>(derived(), startIndices, sizes);
    }
    template <typename StartIndices, typename Sizes> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorSlicingOp<const StartIndices, const Sizes, Derived>
    slice(const StartIndices& startIndices, const Sizes& sizes) {
      return TensorSlicingOp<const StartIndices, const Sizes, Derived>(derived(), startIndices, sizes);
    }

    template <DenseIndex DimId> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorChippingOp<DimId, const Derived>
    chip(const Index offset) const {
      return TensorChippingOp<DimId, const Derived>(derived(), offset, DimId);
    }
    template <Index DimId> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorChippingOp<DimId, Derived>
    chip(const Index offset) {
      return TensorChippingOp<DimId, Derived>(derived(), offset, DimId);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorChippingOp<Dynamic, const Derived>
    chip(const Index offset, const Index dim) const {
      return TensorChippingOp<Dynamic, const Derived>(derived(), offset, dim);
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorChippingOp<Dynamic, Derived>
    chip(const Index offset, const Index dim) {
      return TensorChippingOp<Dynamic, Derived>(derived(), offset, dim);
    }

    template <typename ReverseDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReverseOp<const ReverseDimensions, const Derived>
    reverse(const ReverseDimensions& rev) const {
      return TensorReverseOp<const ReverseDimensions, const Derived>(derived(), rev);
    }
    template <typename ReverseDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorReverseOp<const ReverseDimensions, Derived>
    reverse(const ReverseDimensions& rev) {
      return TensorReverseOp<const ReverseDimensions, Derived>(derived(), rev);
    }

    template <typename Shuffle> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorShufflingOp<const Shuffle, const Derived>
    shuffle(const Shuffle& shuffle) const {
      return TensorShufflingOp<const Shuffle, const Derived>(derived(), shuffle);
    }
    template <typename Shuffle> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorShufflingOp<const Shuffle, Derived>
    shuffle(const Shuffle& shuffle) {
      return TensorShufflingOp<const Shuffle, Derived>(derived(), shuffle);
    }

    template <typename Strides> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorStridingOp<const Strides, const Derived>
    stride(const Strides& strides) const {
      return TensorStridingOp<const Strides, const Derived>(derived(), strides);
    }
    template <typename Strides> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorStridingOp<const Strides, Derived>
    stride(const Strides& strides) {
      return TensorStridingOp<const Strides, Derived>(derived(), strides);
    }

    // Select the device on which to evaluate the expression.
    template <typename DeviceType>
    TensorDevice<Derived, DeviceType> device(const DeviceType& device) {
      return TensorDevice<Derived, DeviceType>(device, derived());
    }

 protected:
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& derived() { return *static_cast<Derived*>(this); }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_BASE_H
