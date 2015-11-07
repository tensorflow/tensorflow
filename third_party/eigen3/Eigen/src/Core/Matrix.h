// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_H
#define EIGEN_MATRIX_H

namespace Eigen {

/** \class Matrix
  * \ingroup Core_Module
  *
  * \brief The matrix class, also used for vectors and row-vectors
  *
  * The %Matrix class is the work-horse for all \em dense (\ref dense "note") matrices and vectors within Eigen.
  * Vectors are matrices with one column, and row-vectors are matrices with one row.
  *
  * The %Matrix class encompasses \em both fixed-size and dynamic-size objects (\ref fixedsize "note").
  *
  * The first three template parameters are required:
  * \tparam _Scalar \anchor matrix_tparam_scalar Numeric type, e.g. float, double, int or std::complex<float>.
  *                 User defined sclar types are supported as well (see \ref user_defined_scalars "here").
  * \tparam _Rows Number of rows, or \b Dynamic
  * \tparam _Cols Number of columns, or \b Dynamic
  *
  * The remaining template parameters are optional -- in most cases you don't have to worry about them.
  * \tparam _Options \anchor matrix_tparam_options A combination of either \b #RowMajor or \b #ColMajor, and of either
  *                 \b #AutoAlign or \b #DontAlign.
  *                 The former controls \ref TopicStorageOrders "storage order", and defaults to column-major. The latter controls alignment, which is required
  *                 for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
  * \tparam _MaxRows Maximum number of rows. Defaults to \a _Rows (\ref maxrows "note").
  * \tparam _MaxCols Maximum number of columns. Defaults to \a _Cols (\ref maxrows "note").
  *
  * Eigen provides a number of typedefs covering the usual cases. Here are some examples:
  *
  * \li \c Matrix2d is a 2x2 square matrix of doubles (\c Matrix<double, 2, 2>)
  * \li \c Vector4f is a vector of 4 floats (\c Matrix<float, 4, 1>)
  * \li \c RowVector3i is a row-vector of 3 ints (\c Matrix<int, 1, 3>)
  *
  * \li \c MatrixXf is a dynamic-size matrix of floats (\c Matrix<float, Dynamic, Dynamic>)
  * \li \c VectorXf is a dynamic-size vector of floats (\c Matrix<float, Dynamic, 1>)
  *
  * \li \c Matrix2Xf is a partially fixed-size (dynamic-size) matrix of floats (\c Matrix<float, 2, Dynamic>)
  * \li \c MatrixX3d is a partially dynamic-size (fixed-size) matrix of double (\c Matrix<double, Dynamic, 3>)
  *
  * See \link matrixtypedefs this page \endlink for a complete list of predefined \em %Matrix and \em Vector typedefs.
  *
  * You can access elements of vectors and matrices using normal subscripting:
  *
  * \code
  * Eigen::VectorXd v(10);
  * v[0] = 0.1;
  * v[1] = 0.2;
  * v(0) = 0.3;
  * v(1) = 0.4;
  *
  * Eigen::MatrixXi m(10, 10);
  * m(0, 1) = 1;
  * m(0, 2) = 2;
  * m(0, 3) = 3;
  * \endcode
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_MATRIX_PLUGIN.
  *
  * <i><b>Some notes:</b></i>
  *
  * <dl>
  * <dt><b>\anchor dense Dense versus sparse:</b></dt>
  * <dd>This %Matrix class handles dense, not sparse matrices and vectors. For sparse matrices and vectors, see the Sparse module.
  *
  * Dense matrices and vectors are plain usual arrays of coefficients. All the coefficients are stored, in an ordinary contiguous array.
  * This is unlike Sparse matrices and vectors where the coefficients are stored as a list of nonzero coefficients.</dd>
  *
  * <dt><b>\anchor fixedsize Fixed-size versus dynamic-size:</b></dt>
  * <dd>Fixed-size means that the numbers of rows and columns are known are compile-time. In this case, Eigen allocates the array
  * of coefficients as a fixed-size array, as a class member. This makes sense for very small matrices, typically up to 4x4, sometimes up
  * to 16x16. Larger matrices should be declared as dynamic-size even if one happens to know their size at compile-time.
  *
  * Dynamic-size means that the numbers of rows or columns are not necessarily known at compile-time. In this case they are runtime
  * variables, and the array of coefficients is allocated dynamically on the heap.
  *
  * Note that \em dense matrices, be they Fixed-size or Dynamic-size, <em>do not</em> expand dynamically in the sense of a std::map.
  * If you want this behavior, see the Sparse module.</dd>
  *
  * <dt><b>\anchor maxrows _MaxRows and _MaxCols:</b></dt>
  * <dd>In most cases, one just leaves these parameters to the default values.
  * These parameters mean the maximum size of rows and columns that the matrix may have. They are useful in cases
  * when the exact numbers of rows and columns are not known are compile-time, but it is known at compile-time that they cannot
  * exceed a certain value. This happens when taking dynamic-size blocks inside fixed-size matrices: in this case _MaxRows and _MaxCols
  * are the dimensions of the original matrix, while _Rows and _Cols are Dynamic.</dd>
  * </dl>
  *
  * \see MatrixBase for the majority of the API methods for matrices, \ref TopicClassHierarchy, 
  * \ref TopicStorageOrders 
  */

namespace internal {
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct traits<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> >
{
  typedef _Scalar Scalar;
  typedef Dense StorageKind;
  typedef DenseIndex Index;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = _Rows,
    ColsAtCompileTime = _Cols,
    MaxRowsAtCompileTime = _MaxRows,
    MaxColsAtCompileTime = _MaxCols,
    Flags = compute_matrix_flags<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::ret,
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    Options = _Options,
    InnerStrideAtCompileTime = 1,
    OuterStrideAtCompileTime = (Options&RowMajor) ? ColsAtCompileTime : RowsAtCompileTime
  };
};
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
class Matrix
  : public PlainObjectBase<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> >
{
  public:

    /** \brief Base class typedef.
      * \sa PlainObjectBase
      */
    typedef PlainObjectBase<Matrix> Base;

    enum { Options = _Options };

    EIGEN_DENSE_PUBLIC_INTERFACE(Matrix)

    typedef typename Base::PlainObject PlainObject;

    using Base::base;
    using Base::coeffRef;

    /**
      * \brief Assigns matrices to each other.
      *
      * \note This is a special case of the templated operator=. Its purpose is
      * to prevent a default operator= from hiding the templated operator=.
      *
      * \callgraph
      */
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix& operator=(const Matrix& other)
    {
      return Base::_set(other);
    }

    /** \internal
      * \brief Copies the value of the expression \a other into \c *this with automatic resizing.
      *
      * *this might be resized to match the dimensions of \a other. If *this was a null matrix (not already initialized),
      * it will be initialized.
      *
      * Note that copying a row-vector into a vector (and conversely) is allowed.
      * The resizing, if any, is then done in the appropriate way so that row-vectors
      * remain row-vectors and vectors remain vectors.
      */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      return Base::_set(other);
    }

    /* Here, doxygen failed to copy the brief information when using \copydoc */

    /**
      * \brief Copies the generic expression \a other into *this.
      * \copydetails DenseBase::operator=(const EigenBase<OtherDerived> &other)
      */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix& operator=(const EigenBase<OtherDerived> &other)
    {
      return Base::operator=(other);
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix& operator=(const ReturnByValue<OtherDerived>& func)
    {
      return Base::operator=(func);
    }

    /** \brief Default constructor.
      *
      * For fixed-size matrices, does nothing.
      *
      * For dynamic-size matrices, creates an empty matrix of size 0. Does not allocate any array. Such a matrix
      * is called a null matrix. This constructor is the unique way to create null matrices: resizing
      * a matrix to 0 is not supported.
      *
      * \sa resize(Index,Index)
      */
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix() : Base()
    {
      Base::_check_template_params();
      EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
    }

    // FIXME is it still needed
    EIGEN_DEVICE_FUNC
    Matrix(internal::constructor_without_unaligned_array_assert)
      : Base(internal::constructor_without_unaligned_array_assert())
    { Base::_check_template_params(); EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED }

#ifdef EIGEN_HAVE_RVALUE_REFERENCES
    Matrix(Matrix&& other)
      : Base(std::move(other))
    {
      Base::_check_template_params();
      if (RowsAtCompileTime!=Dynamic && ColsAtCompileTime!=Dynamic)
        Base::_set_noalias(other);
    }
    Matrix& operator=(Matrix&& other)
    {
      other.swap(*this);
      return *this;
    }
#endif

    #ifndef EIGEN_PARSED_BY_DOXYGEN

    // This constructor is for both 1x1 matrices and dynamic vectors
    template<typename T>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE explicit Matrix(const T& x)
    {
      Base::_check_template_params();
      Base::template _init1<T>(x);
    }

    template<typename T0, typename T1>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix(const T0& x, const T1& y)
    {
      Base::_check_template_params();
      Base::template _init2<T0,T1>(x, y);
    }
    #else
    /** \brief Constructs a fixed-sized matrix initialized with coefficients starting at \a data */
    EIGEN_DEVICE_FUNC
    explicit Matrix(const Scalar *data);

    /** \brief Constructs a vector or row-vector with given dimension. \only_for_vectors
      *
      * Note that this is only useful for dynamic-size vectors. For fixed-size vectors,
      * it is redundant to pass the dimension here, so it makes more sense to use the default
      * constructor Matrix() instead.
      */
    EIGEN_STRONG_INLINE explicit Matrix(Index dim);
    /** \brief Constructs an initialized 1x1 matrix with the given coefficient */
    Matrix(const Scalar& x);
    /** \brief Constructs an uninitialized matrix with \a rows rows and \a cols columns.
      *
      * This is useful for dynamic-size matrices. For fixed-size matrices,
      * it is redundant to pass these parameters, so one should use the default constructor
      * Matrix() instead. */
    EIGEN_DEVICE_FUNC
    Matrix(Index rows, Index cols);
    /** \brief Constructs an initialized 2D vector with given coefficients */
    Matrix(const Scalar& x, const Scalar& y);
    #endif

    /** \brief Constructs an initialized 3D vector with given coefficients */
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix(const Scalar& x, const Scalar& y, const Scalar& z)
    {
      Base::_check_template_params();
      EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 3)
      m_storage.data()[0] = x;
      m_storage.data()[1] = y;
      m_storage.data()[2] = z;
    }
    /** \brief Constructs an initialized 4D vector with given coefficients */
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& w)
    {
      Base::_check_template_params();
      EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 4)
      m_storage.data()[0] = x;
      m_storage.data()[1] = y;
      m_storage.data()[2] = z;
      m_storage.data()[3] = w;
    }


    /** \brief Constructor copying the value of the expression \a other */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix(const MatrixBase<OtherDerived>& other)
             : Base(other.rows() * other.cols(), other.rows(), other.cols())
    {
      // This test resides here, to bring the error messages closer to the user. Normally, these checks
      // are performed deeply within the library, thus causing long and scary error traces.
      EIGEN_STATIC_ASSERT((internal::is_same<Scalar, typename OtherDerived::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

      Base::_check_template_params();
      Base::_set_noalias(other);
    }
    /** \brief Copy constructor */
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix(const Matrix& other)
            : Base(other.rows() * other.cols(), other.rows(), other.cols())
    {
      Base::_check_template_params();
      Base::_set_noalias(other);
    }
    /** \brief Copy constructor with in-place evaluation */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix(const ReturnByValue<OtherDerived>& other)
    {
      Base::_check_template_params();
      Base::resize(other.rows(), other.cols());
      other.evalTo(*this);
    }

    /** \brief Copy constructor for generic expressions.
      * \sa MatrixBase::operator=(const EigenBase<OtherDerived>&)
      */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Matrix(const EigenBase<OtherDerived> &other)
      : Base(other.derived().rows() * other.derived().cols(), other.derived().rows(), other.derived().cols())
    {
      Base::_check_template_params();
      Base::_resize_to_match(other);
      // FIXME/CHECK: isn't *this = other.derived() more efficient. it allows to
      //              go for pure _set() implementations, right?
      *this = other;
    }

    /** \internal
      * \brief Override MatrixBase::swap() since for dynamic-sized matrices
      * of same type it is enough to swap the data pointers.
      */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void swap(MatrixBase<OtherDerived> const & other)
    { this->_swap(other.derived()); }

    EIGEN_DEVICE_FUNC inline Index innerStride() const { return 1; }
    EIGEN_DEVICE_FUNC inline Index outerStride() const { return this->innerSize(); }

    /////////// Geometry module ///////////

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    explicit Matrix(const RotationBase<OtherDerived,ColsAtCompileTime>& r);
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Matrix& operator=(const RotationBase<OtherDerived,ColsAtCompileTime>& r);

    #ifdef EIGEN2_SUPPORT
    template<typename OtherDerived>
    explicit Matrix(const eigen2_RotationBase<OtherDerived,ColsAtCompileTime>& r);
    template<typename OtherDerived>
    Matrix& operator=(const eigen2_RotationBase<OtherDerived,ColsAtCompileTime>& r);
    #endif

    // allow to extend Matrix outside Eigen
    #ifdef EIGEN_MATRIX_PLUGIN
    #include EIGEN_MATRIX_PLUGIN
    #endif

  protected:
    template <typename Derived, typename OtherDerived, bool IsVector>
    friend struct internal::conservative_resize_like_impl;

    using Base::m_storage;
};

/** \defgroup matrixtypedefs Global matrix typedefs
  *
  * \ingroup Core_Module
  *
  * Eigen defines several typedef shortcuts for most common matrix and vector types.
  *
  * The general patterns are the following:
  *
  * \c MatrixSizeType where \c Size can be \c 2,\c 3,\c 4 for fixed size square matrices or \c X for dynamic size,
  * and where \c Type can be \c i for integer, \c f for float, \c d for double, \c cf for complex float, \c cd
  * for complex double.
  *
  * For example, \c Matrix3d is a fixed-size 3x3 matrix type of doubles, and \c MatrixXf is a dynamic-size matrix of floats.
  *
  * There are also \c VectorSizeType and \c RowVectorSizeType which are self-explanatory. For example, \c Vector4cf is
  * a fixed-size vector of 4 complex floats.
  *
  * \sa class Matrix
  */

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, Size> Matrix##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, 1>    Vector##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, 1, Size>    RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)         \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, Dynamic> Matrix##Size##X##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Dynamic, Size> Matrix##X##Size##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Dynamic, X) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(int,                  i)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(float,                f)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(double,               d)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<float>,  cf)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<double>, cd)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

} // end namespace Eigen

#endif // EIGEN_MATRIX_H
