// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN2SUPPORT_H
#define EIGEN2SUPPORT_H

#if (!defined(EIGEN2_SUPPORT)) || (!defined(EIGEN_CORE_H))
#error Eigen2 support must be enabled by defining EIGEN2_SUPPORT before including any Eigen header
#endif

#include "src/Core/util/DisableStupidWarnings.h"

/** \ingroup Support_modules
  * \defgroup Eigen2Support_Module Eigen2 support module
  * This module provides a couple of deprecated functions improving the compatibility with Eigen2.
  *
  * To use it, define EIGEN2_SUPPORT before including any Eigen header
  * \code
  * #define EIGEN2_SUPPORT
  * \endcode
  *
  */

#include "src/Eigen2Support/Macros.h"
#include "src/Eigen2Support/Memory.h"
#include "src/Eigen2Support/Meta.h"
#include "src/Eigen2Support/Lazy.h"
#include "src/Eigen2Support/Cwise.h"
#include "src/Eigen2Support/CwiseOperators.h"
#include "src/Eigen2Support/TriangularSolver.h"
#include "src/Eigen2Support/Block.h"
#include "src/Eigen2Support/VectorBlock.h"
#include "src/Eigen2Support/Minor.h"
#include "src/Eigen2Support/MathFunctions.h"


#include "src/Core/util/ReenableStupidWarnings.h"

// Eigen2 used to include iostream
#include<iostream>

#define EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, SizeSuffix) \
using Eigen::Matrix##SizeSuffix##TypeSuffix; \
using Eigen::Vector##SizeSuffix##TypeSuffix; \
using Eigen::RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(TypeSuffix) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, 2) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, 3) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, 4) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, X) \

#define EIGEN_USING_MATRIX_TYPEDEFS \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(i) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(f) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(d) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(cf) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(cd)

#define USING_PART_OF_NAMESPACE_EIGEN \
EIGEN_USING_MATRIX_TYPEDEFS \
using Eigen::Matrix; \
using Eigen::MatrixBase; \
using Eigen::ei_random; \
using Eigen::ei_real; \
using Eigen::ei_imag; \
using Eigen::ei_conj; \
using Eigen::ei_abs; \
using Eigen::ei_abs2; \
using Eigen::ei_sqrt; \
using Eigen::ei_exp; \
using Eigen::ei_log; \
using Eigen::ei_sin; \
using Eigen::ei_cos;

#endif // EIGEN2SUPPORT_H
