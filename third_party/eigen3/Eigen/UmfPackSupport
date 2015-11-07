#ifndef EIGEN_UMFPACKSUPPORT_MODULE_H
#define EIGEN_UMFPACKSUPPORT_MODULE_H

#include "SparseCore"

#include "src/Core/util/DisableStupidWarnings.h"

extern "C" {
#include <umfpack.h>
}

/** \ingroup Support_modules
  * \defgroup UmfPackSupport_Module UmfPackSupport module
  *
  * This module provides an interface to the UmfPack library which is part of the <a href="http://www.cise.ufl.edu/research/sparse/SuiteSparse/">suitesparse</a> package.
  * It provides the following factorization class:
  * - class UmfPackLU: a multifrontal sequential LU factorization.
  *
  * \code
  * #include <Eigen/UmfPackSupport>
  * \endcode
  *
  * In order to use this module, the umfpack headers must be accessible from the include paths, and your binary must be linked to the umfpack library and its dependencies.
  * The dependencies depend on how umfpack has been compiled.
  * For a cmake based project, you can use our FindUmfPack.cmake module to help you in this task.
  *
  */

#include "src/misc/Solve.h"
#include "src/misc/SparseSolve.h"

#include "src/UmfPackSupport/UmfPackSupport.h"

#include "src/Core/util/ReenableStupidWarnings.h"

#endif // EIGEN_UMFPACKSUPPORT_MODULE_H
