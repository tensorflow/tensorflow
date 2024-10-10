#ifndef NUMPY_CORE_INCLUDE_NUMPY_NOPREFIX_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NOPREFIX_H_

/*
 * You can directly include noprefix.h as a backward
 * compatibility measure
 */
#ifndef NPY_NO_PREFIX
#include "ndarrayobject.h"
#include "npy_interrupt.h"
#endif

#define SIGSETJMP   NPY_SIGSETJMP
#define SIGLONGJMP  NPY_SIGLONGJMP
#define SIGJMP_BUF  NPY_SIGJMP_BUF

#define MAX_DIMS NPY_MAXDIMS

#define longlong    npy_longlong
#define ulonglong   npy_ulonglong
#define Bool        npy_bool
#define longdouble  npy_longdouble
#define byte        npy_byte

#ifndef _BSD_SOURCE
#define ushort      npy_ushort
#define uint        npy_uint
#define ulong       npy_ulong
#endif

#define ubyte       npy_ubyte
#define ushort      npy_ushort
#define uint        npy_uint
#define ulong       npy_ulong
#define cfloat      npy_cfloat
#define cdouble     npy_cdouble
#define clongdouble npy_clongdouble
#define Int8        npy_int8
#define UInt8       npy_uint8
#define Int16       npy_int16
#define UInt16      npy_uint16
#define Int32       npy_int32
#define UInt32      npy_uint32
#define Int64       npy_int64
#define UInt64      npy_uint64
#define Int128      npy_int128
#define UInt128     npy_uint128
#define Int256      npy_int256
#define UInt256     npy_uint256
#define Float16     npy_float16
#define Complex32   npy_complex32
#define Float32     npy_float32
#define Complex64   npy_complex64
#define Float64     npy_float64
#define Complex128  npy_complex128
#define Float80     npy_float80
#define Complex160  npy_complex160
#define Float96     npy_float96
#define Complex192  npy_complex192
#define Float128    npy_float128
#define Complex256  npy_complex256
#define intp        npy_intp
#define uintp       npy_uintp
#define datetime    npy_datetime
#define timedelta   npy_timedelta

#define SIZEOF_LONGLONG         NPY_SIZEOF_LONGLONG
#define SIZEOF_INTP             NPY_SIZEOF_INTP
#define SIZEOF_UINTP            NPY_SIZEOF_UINTP
#define SIZEOF_HALF             NPY_SIZEOF_HALF
#define SIZEOF_LONGDOUBLE       NPY_SIZEOF_LONGDOUBLE
#define SIZEOF_DATETIME         NPY_SIZEOF_DATETIME
#define SIZEOF_TIMEDELTA        NPY_SIZEOF_TIMEDELTA

#define LONGLONG_FMT NPY_LONGLONG_FMT
#define ULONGLONG_FMT NPY_ULONGLONG_FMT
#define LONGLONG_SUFFIX NPY_LONGLONG_SUFFIX
#define ULONGLONG_SUFFIX NPY_ULONGLONG_SUFFIX

#define MAX_INT8 127
#define MIN_INT8 -128
#define MAX_UINT8 255
#define MAX_INT16 32767
#define MIN_INT16 -32768
#define MAX_UINT16 65535
#define MAX_INT32 2147483647
#define MIN_INT32 (-MAX_INT32 - 1)
#define MAX_UINT32 4294967295U
#define MAX_INT64 LONGLONG_SUFFIX(9223372036854775807)
#define MIN_INT64 (-MAX_INT64 - LONGLONG_SUFFIX(1))
#define MAX_UINT64 ULONGLONG_SUFFIX(18446744073709551615)
#define MAX_INT128 LONGLONG_SUFFIX(85070591730234615865843651857942052864)
#define MIN_INT128 (-MAX_INT128 - LONGLONG_SUFFIX(1))
#define MAX_UINT128 ULONGLONG_SUFFIX(170141183460469231731687303715884105728)
#define MAX_INT256 LONGLONG_SUFFIX(57896044618658097711785492504343953926634992332820282019728792003956564819967)
#define MIN_INT256 (-MAX_INT256 - LONGLONG_SUFFIX(1))
#define MAX_UINT256 ULONGLONG_SUFFIX(115792089237316195423570985008687907853269984665640564039457584007913129639935)

#define MAX_BYTE NPY_MAX_BYTE
#define MIN_BYTE NPY_MIN_BYTE
#define MAX_UBYTE NPY_MAX_UBYTE
#define MAX_SHORT NPY_MAX_SHORT
#define MIN_SHORT NPY_MIN_SHORT
#define MAX_USHORT NPY_MAX_USHORT
#define MAX_INT   NPY_MAX_INT
#define MIN_INT   NPY_MIN_INT
#define MAX_UINT  NPY_MAX_UINT
#define MAX_LONG  NPY_MAX_LONG
#define MIN_LONG  NPY_MIN_LONG
#define MAX_ULONG  NPY_MAX_ULONG
#define MAX_LONGLONG NPY_MAX_LONGLONG
#define MIN_LONGLONG NPY_MIN_LONGLONG
#define MAX_ULONGLONG NPY_MAX_ULONGLONG
#define MIN_DATETIME NPY_MIN_DATETIME
#define MAX_DATETIME NPY_MAX_DATETIME
#define MIN_TIMEDELTA NPY_MIN_TIMEDELTA
#define MAX_TIMEDELTA NPY_MAX_TIMEDELTA

#define BITSOF_BOOL       NPY_BITSOF_BOOL
#define BITSOF_CHAR       NPY_BITSOF_CHAR
#define BITSOF_SHORT      NPY_BITSOF_SHORT
#define BITSOF_INT        NPY_BITSOF_INT
#define BITSOF_LONG       NPY_BITSOF_LONG
#define BITSOF_LONGLONG   NPY_BITSOF_LONGLONG
#define BITSOF_HALF       NPY_BITSOF_HALF
#define BITSOF_FLOAT      NPY_BITSOF_FLOAT
#define BITSOF_DOUBLE     NPY_BITSOF_DOUBLE
#define BITSOF_LONGDOUBLE NPY_BITSOF_LONGDOUBLE
#define BITSOF_DATETIME   NPY_BITSOF_DATETIME
#define BITSOF_TIMEDELTA   NPY_BITSOF_TIMEDELTA

#define _pya_malloc PyArray_malloc
#define _pya_free PyArray_free
#define _pya_realloc PyArray_realloc

#define BEGIN_THREADS_DEF NPY_BEGIN_THREADS_DEF
#define BEGIN_THREADS     NPY_BEGIN_THREADS
#define END_THREADS       NPY_END_THREADS
#define ALLOW_C_API_DEF   NPY_ALLOW_C_API_DEF
#define ALLOW_C_API       NPY_ALLOW_C_API
#define DISABLE_C_API     NPY_DISABLE_C_API

#define PY_FAIL NPY_FAIL
#define PY_SUCCEED NPY_SUCCEED

#ifndef TRUE
#define TRUE NPY_TRUE
#endif

#ifndef FALSE
#define FALSE NPY_FALSE
#endif

#define LONGDOUBLE_FMT NPY_LONGDOUBLE_FMT

#define CONTIGUOUS         NPY_CONTIGUOUS
#define C_CONTIGUOUS       NPY_C_CONTIGUOUS
#define FORTRAN            NPY_FORTRAN
#define F_CONTIGUOUS       NPY_F_CONTIGUOUS
#define OWNDATA            NPY_OWNDATA
#define FORCECAST          NPY_FORCECAST
#define ENSURECOPY         NPY_ENSURECOPY
#define ENSUREARRAY        NPY_ENSUREARRAY
#define ELEMENTSTRIDES     NPY_ELEMENTSTRIDES
#define ALIGNED            NPY_ALIGNED
#define NOTSWAPPED         NPY_NOTSWAPPED
#define WRITEABLE          NPY_WRITEABLE
#define WRITEBACKIFCOPY    NPY_ARRAY_WRITEBACKIFCOPY
#define ARR_HAS_DESCR      NPY_ARR_HAS_DESCR
#define BEHAVED            NPY_BEHAVED
#define BEHAVED_NS         NPY_BEHAVED_NS
#define CARRAY             NPY_CARRAY
#define CARRAY_RO          NPY_CARRAY_RO
#define FARRAY             NPY_FARRAY
#define FARRAY_RO          NPY_FARRAY_RO
#define DEFAULT            NPY_DEFAULT
#define IN_ARRAY           NPY_IN_ARRAY
#define OUT_ARRAY          NPY_OUT_ARRAY
#define INOUT_ARRAY        NPY_INOUT_ARRAY
#define IN_FARRAY          NPY_IN_FARRAY
#define OUT_FARRAY         NPY_OUT_FARRAY
#define INOUT_FARRAY       NPY_INOUT_FARRAY
#define UPDATE_ALL         NPY_UPDATE_ALL

#define OWN_DATA          NPY_OWNDATA
#define BEHAVED_FLAGS     NPY_BEHAVED
#define BEHAVED_FLAGS_NS  NPY_BEHAVED_NS
#define CARRAY_FLAGS_RO   NPY_CARRAY_RO
#define CARRAY_FLAGS      NPY_CARRAY
#define FARRAY_FLAGS      NPY_FARRAY
#define FARRAY_FLAGS_RO   NPY_FARRAY_RO
#define DEFAULT_FLAGS     NPY_DEFAULT
#define UPDATE_ALL_FLAGS  NPY_UPDATE_ALL_FLAGS

#ifndef MIN
#define MIN PyArray_MIN
#endif
#ifndef MAX
#define MAX PyArray_MAX
#endif
#define MAX_INTP NPY_MAX_INTP
#define MIN_INTP NPY_MIN_INTP
#define MAX_UINTP NPY_MAX_UINTP
#define INTP_FMT NPY_INTP_FMT

#ifndef PYPY_VERSION
#define REFCOUNT PyArray_REFCOUNT
#define MAX_ELSIZE NPY_MAX_ELSIZE
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NOPREFIX_H_ */
