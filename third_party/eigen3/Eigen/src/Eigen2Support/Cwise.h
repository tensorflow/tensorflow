// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CWISE_H
#define EIGEN_CWISE_H

namespace Eigen { 

/** \internal
  * convenient macro to defined the return type of a cwise binary operation */
#define EIGEN_CWISE_BINOP_RETURN_TYPE(OP) \
    CwiseBinaryOp<OP<typename internal::traits<ExpressionType>::Scalar>, ExpressionType, OtherDerived>

/** \internal
  * convenient macro to defined the return type of a cwise unary operation */
#define EIGEN_CWISE_UNOP_RETURN_TYPE(OP) \
    CwiseUnaryOp<OP<typename internal::traits<ExpressionType>::Scalar>, ExpressionType>

/** \internal
  * convenient macro to defined the return type of a cwise comparison to a scalar */
#define EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(OP) \
    CwiseBinaryOp<OP<typename internal::traits<ExpressionType>::Scalar>, ExpressionType, \
        typename ExpressionType::ConstantReturnType >

/** \class Cwise
  *
  * \brief Pseudo expression providing additional coefficient-wise operations
  *
  * \param ExpressionType the type of the object on which to do coefficient-wise operations
  *
  * This class represents an expression with additional coefficient-wise features.
  * It is the return type of MatrixBase::cwise()
  * and most of the time this is the only way it is used.
  *
  * Example: \include MatrixBase_cwise_const.cpp
  * Output: \verbinclude MatrixBase_cwise_const.out
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_CWISE_PLUGIN.
  *
  * \sa MatrixBase::cwise() const, MatrixBase::cwise()
  */
template<typename ExpressionType> class Cwise
{
  public:

    typedef typename internal::traits<ExpressionType>::Scalar Scalar;
    typedef typename internal::conditional<internal::must_nest_by_value<ExpressionType>::ret,
        ExpressionType, const ExpressionType&>::type ExpressionTypeNested;
    typedef CwiseUnaryOp<internal::scalar_add_op<Scalar>, ExpressionType> ScalarAddReturnType;

    inline Cwise(const ExpressionType& matrix) : m_matrix(matrix) {}

    /** \internal */
    inline const ExpressionType& _expression() const { return m_matrix; }

    template<typename OtherDerived>
    const EIGEN_CWISE_PRODUCT_RETURN_TYPE(ExpressionType,OtherDerived)
    operator*(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    const EIGEN_CWISE_BINOP_RETURN_TYPE(internal::scalar_quotient_op)
    operator/(const MatrixBase<OtherDerived> &other) const;

    /** \deprecated ArrayBase::min() */
    template<typename OtherDerived>
    const EIGEN_CWISE_BINOP_RETURN_TYPE(internal::scalar_min_op)
    (min)(const MatrixBase<OtherDerived> &other) const
    { return EIGEN_CWISE_BINOP_RETURN_TYPE(internal::scalar_min_op)(_expression(), other.derived()); }

    /** \deprecated ArrayBase::max() */
    template<typename OtherDerived>
    const EIGEN_CWISE_BINOP_RETURN_TYPE(internal::scalar_max_op)
    (max)(const MatrixBase<OtherDerived> &other) const
    { return EIGEN_CWISE_BINOP_RETURN_TYPE(internal::scalar_max_op)(_expression(), other.derived()); }

    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_abs_op)      abs() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_abs2_op)     abs2() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_square_op)   square() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_cube_op)     cube() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_inverse_op)  inverse() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_sqrt_op)     sqrt() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_exp_op)      exp() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_log_op)      log() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_cos_op)      cos() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_sin_op)      sin() const;
    const EIGEN_CWISE_UNOP_RETURN_TYPE(internal::scalar_pow_op)      pow(const Scalar& exponent) const;

    const ScalarAddReturnType
    operator+(const Scalar& scalar) const;

    /** \relates Cwise */
    friend const ScalarAddReturnType
    operator+(const Scalar& scalar, const Cwise& mat)
    { return mat + scalar; }

    ExpressionType& operator+=(const Scalar& scalar);

    const ScalarAddReturnType
    operator-(const Scalar& scalar) const;

    ExpressionType& operator-=(const Scalar& scalar);

    template<typename OtherDerived>
    inline ExpressionType& operator*=(const MatrixBase<OtherDerived> &other);

    template<typename OtherDerived>
    inline ExpressionType& operator/=(const MatrixBase<OtherDerived> &other);

    template<typename OtherDerived> const EIGEN_CWISE_BINOP_RETURN_TYPE(std::less)
    operator<(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const EIGEN_CWISE_BINOP_RETURN_TYPE(std::less_equal)
    operator<=(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater)
    operator>(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater_equal)
    operator>=(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const EIGEN_CWISE_BINOP_RETURN_TYPE(std::equal_to)
    operator==(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const EIGEN_CWISE_BINOP_RETURN_TYPE(std::not_equal_to)
    operator!=(const MatrixBase<OtherDerived>& other) const;

    // comparisons to a scalar value
    const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::less)
    operator<(Scalar s) const;

    const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::less_equal)
    operator<=(Scalar s) const;

    const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::greater)
    operator>(Scalar s) const;

    const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::greater_equal)
    operator>=(Scalar s) const;

    const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::equal_to)
    operator==(Scalar s) const;

    const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::not_equal_to)
    operator!=(Scalar s) const;

    // allow to extend Cwise outside Eigen
    #ifdef EIGEN_CWISE_PLUGIN
    #include EIGEN_CWISE_PLUGIN
    #endif

  protected:
    ExpressionTypeNested m_matrix;
};


/** \returns a Cwise wrapper of *this providing additional coefficient-wise operations
  *
  * Example: \include MatrixBase_cwise_const.cpp
  * Output: \verbinclude MatrixBase_cwise_const.out
  *
  * \sa class Cwise, cwise()
  */
template<typename Derived>
inline const Cwise<Derived> MatrixBase<Derived>::cwise() const
{
  return derived();
}

/** \returns a Cwise wrapper of *this providing additional coefficient-wise operations
  *
  * Example: \include MatrixBase_cwise.cpp
  * Output: \verbinclude MatrixBase_cwise.out
  *
  * \sa class Cwise, cwise() const
  */
template<typename Derived>
inline Cwise<Derived> MatrixBase<Derived>::cwise()
{
  return derived();
}

} // end namespace Eigen

#endif // EIGEN_CWISE_H
