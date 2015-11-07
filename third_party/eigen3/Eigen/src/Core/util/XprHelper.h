// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_XPRHELPER_H
#define EIGEN_XPRHELPER_H

// just a workaround because GCC seems to not really like empty structs
// FIXME: gcc 4.3 generates bad code when strict-aliasing is enabled
// so currently we simply disable this optimization for gcc 4.3
#if EIGEN_COMP_GNUC && !EIGEN_GNUC_AT(4,3)
  #define EIGEN_EMPTY_STRUCT_CTOR(X) \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE X() {} \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE X(const X& ) {}
#else
  #define EIGEN_EMPTY_STRUCT_CTOR(X)
#endif

namespace Eigen {

typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE DenseIndex;

namespace internal {

//classes inheriting no_assignment_operator don't generate a default operator=.
class no_assignment_operator
{
  private:
    no_assignment_operator& operator=(const no_assignment_operator&);
};

/** \internal return the index type with the largest number of bits */
template<typename I1, typename I2>
struct promote_index_type
{
  typedef typename conditional<(sizeof(I1)<sizeof(I2)), I2, I1>::type type;
};

/** \internal If the template parameter Value is Dynamic, this class is just a wrapper around a T variable that
  * can be accessed using value() and setValue().
  * Otherwise, this class is an empty structure and value() just returns the template parameter Value.
  */
template<typename T, int Value> class variable_if_dynamic
{
  public:
    EIGEN_EMPTY_STRUCT_CTOR(variable_if_dynamic)
    EIGEN_DEVICE_FUNC explicit variable_if_dynamic(T v) { EIGEN_ONLY_USED_FOR_DEBUG(v); eigen_assert(v == T(Value)); }
    EIGEN_DEVICE_FUNC static T value() { return T(Value); }
    EIGEN_DEVICE_FUNC void setValue(T) {}
};

template<typename T> class variable_if_dynamic<T, Dynamic>
{
    T m_value;
    EIGEN_DEVICE_FUNC variable_if_dynamic() { eigen_assert(false); }
  public:
    EIGEN_DEVICE_FUNC explicit variable_if_dynamic(T value) : m_value(value) {}
    EIGEN_DEVICE_FUNC T value() const { return m_value; }
    EIGEN_DEVICE_FUNC void setValue(T value) { m_value = value; }
};

/** \internal like variable_if_dynamic but for DynamicIndex
  */
template<typename T, int Value> class variable_if_dynamicindex
{
  public:
    EIGEN_EMPTY_STRUCT_CTOR(variable_if_dynamicindex)
    EIGEN_DEVICE_FUNC explicit variable_if_dynamicindex(T v) { EIGEN_ONLY_USED_FOR_DEBUG(v); eigen_assert(v == T(Value)); }
    EIGEN_DEVICE_FUNC static T value() { return T(Value); }
    EIGEN_DEVICE_FUNC void setValue(T) {}
};

template<typename T> class variable_if_dynamicindex<T, DynamicIndex>
{
    T m_value;
    EIGEN_DEVICE_FUNC variable_if_dynamicindex() { eigen_assert(false); }
  public:
    EIGEN_DEVICE_FUNC explicit variable_if_dynamicindex(T value) : m_value(value) {}
    EIGEN_DEVICE_FUNC T value() const { return m_value; }
    EIGEN_DEVICE_FUNC void setValue(T value) { m_value = value; }
};

template<typename T> struct functor_traits
{
  enum
  {
    Cost = 10,
    PacketAccess = false,
    IsRepeatable = false
  };
};

template<typename T> struct packet_traits;

template<typename T> struct unpacket_traits
{
  typedef T type;
  typedef T half;
  enum {size=1};
};

template<typename _Scalar, int _Rows, int _Cols,
         int _Options = AutoAlign |
                          ( (_Rows==1 && _Cols!=1) ? RowMajor
                          : (_Cols==1 && _Rows!=1) ? ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
         int _MaxRows = _Rows,
         int _MaxCols = _Cols
> class make_proper_matrix_type
{
    enum {
      IsColVector = _Cols==1 && _Rows!=1,
      IsRowVector = _Rows==1 && _Cols!=1,
      Options = IsColVector ? (_Options | ColMajor) & ~RowMajor
              : IsRowVector ? (_Options | RowMajor) & ~ColMajor
              : _Options
    };
  public:
    typedef Matrix<_Scalar, _Rows, _Cols, Options, _MaxRows, _MaxCols> type;
};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class compute_matrix_flags
{
    enum {
      row_major_bit = Options&RowMajor ? RowMajorBit : 0,
      is_dynamic_size_storage = MaxRows==Dynamic || MaxCols==Dynamic,

      aligned_bit =
      (
            ((Options&DontAlign)==0)
        && (
#if EIGEN_ALIGN_STATICALLY
             ((!is_dynamic_size_storage) && (((MaxCols*MaxRows*int(sizeof(Scalar))) % EIGEN_ALIGN_BYTES) == 0))
#else
             0
#endif

          ||

#if EIGEN_ALIGN
             is_dynamic_size_storage
#else
             0
#endif

          )
      ) ? AlignedBit : 0,
      packet_access_bit = packet_traits<Scalar>::Vectorizable && aligned_bit ? PacketAccessBit : 0
    };

  public:
    enum { ret = LinearAccessBit | LvalueBit | DirectAccessBit | NestByRefBit | packet_access_bit | row_major_bit | aligned_bit };
};

template<int _Rows, int _Cols> struct size_at_compile_time
{
  enum { ret = (_Rows==Dynamic || _Cols==Dynamic) ? Dynamic : _Rows * _Cols };
};

/* plain_matrix_type : the difference from eval is that plain_matrix_type is always a plain matrix type,
 * whereas eval is a const reference in the case of a matrix
 */

template<typename T, typename StorageKind = typename traits<T>::StorageKind> struct plain_matrix_type;
template<typename T, typename BaseClassType> struct plain_matrix_type_dense;
template<typename T> struct plain_matrix_type<T,Dense>
{
  typedef typename plain_matrix_type_dense<T,typename traits<T>::XprKind>::type type;
};

template<typename T> struct plain_matrix_type_dense<T,MatrixXpr>
{
  typedef Matrix<typename traits<T>::Scalar,
                traits<T>::RowsAtCompileTime,
                traits<T>::ColsAtCompileTime,
                AutoAlign | (traits<T>::Flags&RowMajorBit ? RowMajor : ColMajor),
                traits<T>::MaxRowsAtCompileTime,
                traits<T>::MaxColsAtCompileTime
          > type;
};

template<typename T> struct plain_matrix_type_dense<T,ArrayXpr>
{
  typedef Array<typename traits<T>::Scalar,
                traits<T>::RowsAtCompileTime,
                traits<T>::ColsAtCompileTime,
                AutoAlign | (traits<T>::Flags&RowMajorBit ? RowMajor : ColMajor),
                traits<T>::MaxRowsAtCompileTime,
                traits<T>::MaxColsAtCompileTime
          > type;
};

/* eval : the return type of eval(). For matrices, this is just a const reference
 * in order to avoid a useless copy
 */

template<typename T, typename StorageKind = typename traits<T>::StorageKind> struct eval;

template<typename T> struct eval<T,Dense>
{
  typedef typename plain_matrix_type<T>::type type;
//   typedef typename T::PlainObject type;
//   typedef T::Matrix<typename traits<T>::Scalar,
//                 traits<T>::RowsAtCompileTime,
//                 traits<T>::ColsAtCompileTime,
//                 AutoAlign | (traits<T>::Flags&RowMajorBit ? RowMajor : ColMajor),
//                 traits<T>::MaxRowsAtCompileTime,
//                 traits<T>::MaxColsAtCompileTime
//           > type;
};

// for matrices, no need to evaluate, just use a const reference to avoid a useless copy
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct eval<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Dense>
{
  typedef const Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& type;
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct eval<Array<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Dense>
{
  typedef const Array<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& type;
};



/* plain_matrix_type_column_major : same as plain_matrix_type but guaranteed to be column-major
 */
template<typename T> struct plain_matrix_type_column_major
{
  enum { Rows = traits<T>::RowsAtCompileTime,
         Cols = traits<T>::ColsAtCompileTime,
         MaxRows = traits<T>::MaxRowsAtCompileTime,
         MaxCols = traits<T>::MaxColsAtCompileTime
  };
  typedef Matrix<typename traits<T>::Scalar,
                Rows,
                Cols,
                (MaxRows==1&&MaxCols!=1) ? RowMajor : ColMajor,
                MaxRows,
                MaxCols
          > type;
};

/* plain_matrix_type_row_major : same as plain_matrix_type but guaranteed to be row-major
 */
template<typename T> struct plain_matrix_type_row_major
{
  enum { Rows = traits<T>::RowsAtCompileTime,
         Cols = traits<T>::ColsAtCompileTime,
         MaxRows = traits<T>::MaxRowsAtCompileTime,
         MaxCols = traits<T>::MaxColsAtCompileTime
  };
  typedef Matrix<typename traits<T>::Scalar,
                Rows,
                Cols,
                (MaxCols==1&&MaxRows!=1) ? RowMajor : ColMajor,
                MaxRows,
                MaxCols
          > type;
};

// we should be able to get rid of this one too
template<typename T> struct must_nest_by_value { enum { ret = false }; };

/** \internal The reference selector for template expressions. The idea is that we don't
  * need to use references for expressions since they are light weight proxy
  * objects which should generate no copying overhead. */
template <typename T>
struct ref_selector
{
  typedef typename conditional<
    bool(traits<T>::Flags & NestByRefBit),
    T const&,
    const T
  >::type type;
};

/** \internal Adds the const qualifier on the value-type of T2 if and only if T1 is a const type */
template<typename T1, typename T2>
struct transfer_constness
{
  typedef typename conditional<
    bool(internal::is_const<T1>::value),
    typename internal::add_const_on_value_type<T2>::type,
    T2
  >::type type;
};

/** \internal Determines how a given expression should be nested into another one.
  * For example, when you do a * (b+c), Eigen will determine how the expression b+c should be
  * nested into the bigger product expression. The choice is between nesting the expression b+c as-is, or
  * evaluating that expression b+c into a temporary variable d, and nest d so that the resulting expression is
  * a*d. Evaluating can be beneficial for example if every coefficient access in the resulting expression causes
  * many coefficient accesses in the nested expressions -- as is the case with matrix product for example.
  *
  * \param T the type of the expression being nested
  * \param n the number of coefficient accesses in the nested expression for each coefficient access in the bigger expression.
  *
  * Note that if no evaluation occur, then the constness of T is preserved.
  *
  * Example. Suppose that a, b, and c are of type Matrix3d. The user forms the expression a*(b+c).
  * b+c is an expression "sum of matrices", which we will denote by S. In order to determine how to nest it,
  * the Product expression uses: nested<S, 3>::type, which turns out to be Matrix3d because the internal logic of
  * nested determined that in this case it was better to evaluate the expression b+c into a temporary. On the other hand,
  * since a is of type Matrix3d, the Product expression nests it as nested<Matrix3d, 3>::type, which turns out to be
  * const Matrix3d&, because the internal logic of nested determined that since a was already a matrix, there was no point
  * in copying it into another matrix.
  */
template<typename T, int n=1, typename PlainObject = typename eval<T>::type> struct nested
{
  enum {
    // for the purpose of this test, to keep it reasonably simple, we arbitrarily choose a value of Dynamic values.
    // the choice of 10000 makes it larger than any practical fixed value and even most dynamic values.
    // in extreme cases where these assumptions would be wrong, we would still at worst suffer performance issues
    // (poor choice of temporaries).
    // it's important that this value can still be squared without integer overflowing.
    DynamicAsInteger = 10000,
    ScalarReadCost = NumTraits<typename traits<T>::Scalar>::ReadCost,
    ScalarReadCostAsInteger = ScalarReadCost == Dynamic ? int(DynamicAsInteger) : int(ScalarReadCost),
    CoeffReadCost = traits<T>::CoeffReadCost,
    CoeffReadCostAsInteger = CoeffReadCost == Dynamic ? int(DynamicAsInteger) : int(CoeffReadCost),
    NAsInteger = n == Dynamic ? int(DynamicAsInteger) : n,
    CostEvalAsInteger   = (NAsInteger+1) * ScalarReadCostAsInteger + CoeffReadCostAsInteger,
    CostNoEvalAsInteger = NAsInteger * CoeffReadCostAsInteger
  };

  typedef typename conditional<
      ( (int(traits<T>::Flags) & EvalBeforeNestingBit) ||
        int(CostEvalAsInteger) < int(CostNoEvalAsInteger)
      ),
      PlainObject,
      typename ref_selector<T>::type
  >::type type;
};

template<typename T>
EIGEN_DEVICE_FUNC
T* const_cast_ptr(const T* ptr)
{
  return const_cast<T*>(ptr);
}

template<typename Derived, typename XprKind = typename traits<Derived>::XprKind>
struct dense_xpr_base
{
  /* dense_xpr_base should only ever be used on dense expressions, thus falling either into the MatrixXpr or into the ArrayXpr cases */
};

template<typename Derived>
struct dense_xpr_base<Derived, MatrixXpr>
{
  typedef MatrixBase<Derived> type;
};

template<typename Derived>
struct dense_xpr_base<Derived, ArrayXpr>
{
  typedef ArrayBase<Derived> type;
};

/** \internal Helper base class to add a scalar multiple operator
  * overloads for complex types */
template<typename Derived,typename Scalar,typename OtherScalar,
         bool EnableIt = !is_same<Scalar,OtherScalar>::value >
struct special_scalar_op_base : public DenseCoeffsBase<Derived>
{
  // dummy operator* so that the
  // "using special_scalar_op_base::operator*" compiles
  void operator*() const;
};

template<typename Derived,typename Scalar,typename OtherScalar>
struct special_scalar_op_base<Derived,Scalar,OtherScalar,true>  : public DenseCoeffsBase<Derived>
{
  const CwiseUnaryOp<scalar_multiple2_op<Scalar,OtherScalar>, Derived>
  operator*(const OtherScalar& scalar) const
  {
    return CwiseUnaryOp<scalar_multiple2_op<Scalar,OtherScalar>, Derived>
      (*static_cast<const Derived*>(this), scalar_multiple2_op<Scalar,OtherScalar>(scalar));
  }

  inline friend const CwiseUnaryOp<scalar_multiple2_op<Scalar,OtherScalar>, Derived>
  operator*(const OtherScalar& scalar, const Derived& matrix)
  { return static_cast<const special_scalar_op_base&>(matrix).operator*(scalar); }
};

template<typename XprType, typename CastType> struct cast_return_type
{
  typedef typename XprType::Scalar CurrentScalarType;
  typedef typename remove_all<CastType>::type _CastType;
  typedef typename _CastType::Scalar NewScalarType;
  typedef typename conditional<is_same<CurrentScalarType,NewScalarType>::value,
                              const XprType&,CastType>::type type;
};

template <typename A, typename B> struct promote_storage_type;

template <typename A> struct promote_storage_type<A,A>
{
  typedef A ret;
};
template <typename A> struct promote_storage_type<A, const A>
{
  typedef A ret;
};
template <typename A> struct promote_storage_type<const A, A>
{
  typedef A ret;
};



/** \internal gives the plain matrix or array type to store a row/column/diagonal of a matrix type.
  * \param Scalar optional parameter allowing to pass a different scalar type than the one of the MatrixType.
  */
template<typename ExpressionType, typename Scalar = typename ExpressionType::Scalar>
struct plain_row_type
{
  typedef Matrix<Scalar, 1, ExpressionType::ColsAtCompileTime,
                 ExpressionType::PlainObject::Options | RowMajor, 1, ExpressionType::MaxColsAtCompileTime> MatrixRowType;
  typedef Array<Scalar, 1, ExpressionType::ColsAtCompileTime,
                 ExpressionType::PlainObject::Options | RowMajor, 1, ExpressionType::MaxColsAtCompileTime> ArrayRowType;

  typedef typename conditional<
    is_same< typename traits<ExpressionType>::XprKind, MatrixXpr >::value,
    MatrixRowType,
    ArrayRowType 
  >::type type;
};

template<typename ExpressionType, typename Scalar = typename ExpressionType::Scalar>
struct plain_col_type
{
  typedef Matrix<Scalar, ExpressionType::RowsAtCompileTime, 1,
                 ExpressionType::PlainObject::Options & ~RowMajor, ExpressionType::MaxRowsAtCompileTime, 1> MatrixColType;
  typedef Array<Scalar, ExpressionType::RowsAtCompileTime, 1,
                 ExpressionType::PlainObject::Options & ~RowMajor, ExpressionType::MaxRowsAtCompileTime, 1> ArrayColType;

  typedef typename conditional<
    is_same< typename traits<ExpressionType>::XprKind, MatrixXpr >::value,
    MatrixColType,
    ArrayColType 
  >::type type;
};

template<typename ExpressionType, typename Scalar = typename ExpressionType::Scalar>
struct plain_diag_type
{
  enum { diag_size = EIGEN_SIZE_MIN_PREFER_DYNAMIC(ExpressionType::RowsAtCompileTime, ExpressionType::ColsAtCompileTime),
         max_diag_size = EIGEN_SIZE_MIN_PREFER_FIXED(ExpressionType::MaxRowsAtCompileTime, ExpressionType::MaxColsAtCompileTime)
  };
  typedef Matrix<Scalar, diag_size, 1, ExpressionType::PlainObject::Options & ~RowMajor, max_diag_size, 1> MatrixDiagType;
  typedef Array<Scalar, diag_size, 1, ExpressionType::PlainObject::Options & ~RowMajor, max_diag_size, 1> ArrayDiagType;

  typedef typename conditional<
    is_same< typename traits<ExpressionType>::XprKind, MatrixXpr >::value,
    MatrixDiagType,
    ArrayDiagType 
  >::type type;
};

template<typename ExpressionType>
struct is_lvalue
{
  enum { value = !bool(is_const<ExpressionType>::value) &&
                 bool(traits<ExpressionType>::Flags & LvalueBit) };
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_XPRHELPER_H
