#ifndef EIGEN_PASTIXSUPPORT_MODULE_H
#define EIGEN_PASTIXSUPPORT_MODULE_H

#include "SparseCore"

#include "src/Core/util/DisableStupidWarnings.h"

#include <complex.h>
extern "C" {
#include <pastix_nompi.h>
#include <pastix.h>
}

#ifdef complex
#undef complex
#endif

/** \ingroup Support_modules
  * \defgroup PaStiXSupport_Module PaStiXSupport module
  * 
  * This module provides an interface to the <a href="http://pastix.gforge.inria.fr/">PaSTiX</a> library.
  * PaSTiX is a general \b supernodal, \b parallel and \b opensource sparse solver.
  * It provides the two following main factorization classes:
  * - class PastixLLT : a supernodal, parallel LLt Cholesky factorization.
  * - class PastixLDLT: a supernodal, parallel LDLt Cholesky factorization.
  * - class PastixLU : a supernodal, parallel LU factorization (optimized for a symmetric pattern).
  * 
  * \code
  * #include <Eigen/PaStiXSupport>
  * \endcode
  *
  * In order to use this module, the PaSTiX headers must be accessible from the include paths, and your binary must be linked to the PaSTiX library and its dependencies.
  * The dependencies depend on how PaSTiX has been compiled.
  * For a cmake based project, you can use our FindPaSTiX.cmake module to help you in this task.
  *
  */

#include "src/misc/Solve.h"
#include "src/misc/SparseSolve.h"

#include "src/PaStiXSupport/PaStiXSupport.h"


#include "src/Core/util/ReenableStupidWarnings.h"

#endif // EIGEN_PASTIXSUPPORT_MODULE_H
