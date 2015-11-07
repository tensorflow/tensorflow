#ifndef EIGEN_SVD_MODULE_H
#define EIGEN_SVD_MODULE_H

#include "QR"
#include "Householder"
#include "Jacobi"

#include "src/Core/util/DisableStupidWarnings.h"

/** \defgroup SVD_Module SVD module
  *
  *
  *
  * This module provides SVD decomposition for matrices (both real and complex).
  * This decomposition is accessible via the following MatrixBase method:
  *  - MatrixBase::jacobiSvd()
  *
  * \code
  * #include <Eigen/SVD>
  * \endcode
  */

#include "src/misc/Solve.h"
#include "src/SVD/JacobiSVD.h"
#if defined(EIGEN_USE_LAPACKE) && !defined(EIGEN_USE_LAPACKE_STRICT)
#include "src/SVD/JacobiSVD_MKL.h"
#endif
#include "src/SVD/UpperBidiagonalization.h"

#ifdef EIGEN2_SUPPORT
#include "src/Eigen2Support/SVD.h"
#endif

#include "src/Core/util/ReenableStupidWarnings.h"

#endif // EIGEN_SVD_MODULE_H
/* vim: set filetype=cpp et sw=2 ts=2 ai: */
