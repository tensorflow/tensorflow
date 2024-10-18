#ifndef NUMPY_CORE_INCLUDE_NUMPY_OLDNUMERIC_H_
#define NUMPY_CORE_INCLUDE_NUMPY_OLDNUMERIC_H_

/* FIXME -- this file can be deleted? */

#include "arrayobject.h"

#ifndef PYPY_VERSION
#ifndef REFCOUNT
#  define REFCOUNT NPY_REFCOUNT
#  define MAX_ELSIZE 16
#endif
#endif

#define PyArray_UNSIGNED_TYPES
#define PyArray_SBYTE NPY_BYTE
#define PyArray_CopyArray PyArray_CopyInto
#define _PyArray_multiply_list PyArray_MultiplyIntList
#define PyArray_ISSPACESAVER(m) NPY_FALSE
#define PyScalarArray_Check PyArray_CheckScalar

#define CONTIGUOUS NPY_CONTIGUOUS
#define OWN_DIMENSIONS 0
#define OWN_STRIDES 0
#define OWN_DATA NPY_OWNDATA
#define SAVESPACE 0
#define SAVESPACEBIT 0

#undef import_array
#define import_array() { if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); } }

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_OLDNUMERIC_H_ */
