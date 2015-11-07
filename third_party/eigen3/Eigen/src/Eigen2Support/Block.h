// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLOCK2_H
#define EIGEN_BLOCK2_H

namespace Eigen { 

/** \returns a dynamic-size expression of a corner of *this.
  *
  * \param type the type of corner. Can be \a Eigen::TopLeft, \a Eigen::TopRight,
  * \a Eigen::BottomLeft, \a Eigen::BottomRight.
  * \param cRows the number of rows in the corner
  * \param cCols the number of columns in the corner
  *
  * Example: \include MatrixBase_corner_enum_int_int.cpp
  * Output: \verbinclude MatrixBase_corner_enum_int_int.out
  *
  * \note Even though the returned expression has dynamic size, in the case
  * when it is applied to a fixed-size matrix, it inherits a fixed maximal size,
  * which means that evaluating it does not cause a dynamic memory allocation.
  *
  * \sa class Block, block(Index,Index,Index,Index)
  */
template<typename Derived>
inline Block<Derived> DenseBase<Derived>
  ::corner(CornerType type, Index cRows, Index cCols)
{
  switch(type)
  {
    default:
      eigen_assert(false && "Bad corner type.");
    case TopLeft:
      return Block<Derived>(derived(), 0, 0, cRows, cCols);
    case TopRight:
      return Block<Derived>(derived(), 0, cols() - cCols, cRows, cCols);
    case BottomLeft:
      return Block<Derived>(derived(), rows() - cRows, 0, cRows, cCols);
    case BottomRight:
      return Block<Derived>(derived(), rows() - cRows, cols() - cCols, cRows, cCols);
  }
}

/** This is the const version of corner(CornerType, Index, Index).*/
template<typename Derived>
inline const Block<Derived>
DenseBase<Derived>::corner(CornerType type, Index cRows, Index cCols) const
{
  switch(type)
  {
    default:
      eigen_assert(false && "Bad corner type.");
    case TopLeft:
      return Block<Derived>(derived(), 0, 0, cRows, cCols);
    case TopRight:
      return Block<Derived>(derived(), 0, cols() - cCols, cRows, cCols);
    case BottomLeft:
      return Block<Derived>(derived(), rows() - cRows, 0, cRows, cCols);
    case BottomRight:
      return Block<Derived>(derived(), rows() - cRows, cols() - cCols, cRows, cCols);
  }
}

/** \returns a fixed-size expression of a corner of *this.
  *
  * \param type the type of corner. Can be \a Eigen::TopLeft, \a Eigen::TopRight,
  * \a Eigen::BottomLeft, \a Eigen::BottomRight.
  *
  * The template parameters CRows and CCols arethe number of rows and columns in the corner.
  *
  * Example: \include MatrixBase_template_int_int_corner_enum.cpp
  * Output: \verbinclude MatrixBase_template_int_int_corner_enum.out
  *
  * \sa class Block, block(Index,Index,Index,Index)
  */
template<typename Derived>
template<int CRows, int CCols>
inline Block<Derived, CRows, CCols>
DenseBase<Derived>::corner(CornerType type)
{
  switch(type)
  {
    default:
      eigen_assert(false && "Bad corner type.");
    case TopLeft:
      return Block<Derived, CRows, CCols>(derived(), 0, 0);
    case TopRight:
      return Block<Derived, CRows, CCols>(derived(), 0, cols() - CCols);
    case BottomLeft:
      return Block<Derived, CRows, CCols>(derived(), rows() - CRows, 0);
    case BottomRight:
      return Block<Derived, CRows, CCols>(derived(), rows() - CRows, cols() - CCols);
  }
}

/** This is the const version of corner<int, int>(CornerType).*/
template<typename Derived>
template<int CRows, int CCols>
inline const Block<Derived, CRows, CCols>
DenseBase<Derived>::corner(CornerType type) const
{
  switch(type)
  {
    default:
      eigen_assert(false && "Bad corner type.");
    case TopLeft:
      return Block<Derived, CRows, CCols>(derived(), 0, 0);
    case TopRight:
      return Block<Derived, CRows, CCols>(derived(), 0, cols() - CCols);
    case BottomLeft:
      return Block<Derived, CRows, CCols>(derived(), rows() - CRows, 0);
    case BottomRight:
      return Block<Derived, CRows, CCols>(derived(), rows() - CRows, cols() - CCols);
  }
}

} // end namespace Eigen

#endif // EIGEN_BLOCK2_H
