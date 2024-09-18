/*
 * This API is only provided because it is part of publicly exported
 * headers. Its use is considered DEPRECATED, and it will be removed
 * eventually.
 * (This includes the _PyArray_SigintHandler and _PyArray_GetSigintBuf
 * functions which are however, public API, and not headers.)
 *
 * Instead of using these non-threadsafe macros consider periodically
 * querying `PyErr_CheckSignals()` or `PyOS_InterruptOccurred()` will work.
 * Both of these require holding the GIL, although cpython could add a
 * version of `PyOS_InterruptOccurred()` which does not. Such a version
 * actually exists as private API in Python 3.10, and backported to 3.9 and 3.8,
 * see also https://bugs.python.org/issue41037 and
 * https://github.com/python/cpython/pull/20599).
 */

#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_INTERRUPT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_INTERRUPT_H_

#ifndef NPY_NO_SIGNAL

#include <setjmp.h>
#include <signal.h>

#ifndef sigsetjmp

#define NPY_SIGSETJMP(arg1, arg2) setjmp(arg1)
#define NPY_SIGLONGJMP(arg1, arg2) longjmp(arg1, arg2)
#define NPY_SIGJMP_BUF jmp_buf

#else

#define NPY_SIGSETJMP(arg1, arg2) sigsetjmp(arg1, arg2)
#define NPY_SIGLONGJMP(arg1, arg2) siglongjmp(arg1, arg2)
#define NPY_SIGJMP_BUF sigjmp_buf

#endif

#    define NPY_SIGINT_ON {                                             \
                   PyOS_sighandler_t _npy_sig_save;                     \
                   _npy_sig_save = PyOS_setsig(SIGINT, _PyArray_SigintHandler); \
                   if (NPY_SIGSETJMP(*((NPY_SIGJMP_BUF *)_PyArray_GetSigintBuf()), \
                                 1) == 0) {                             \

#    define NPY_SIGINT_OFF }                                      \
        PyOS_setsig(SIGINT, _npy_sig_save);                       \
        }

#else  /* NPY_NO_SIGNAL  */

#define NPY_SIGINT_ON
#define NPY_SIGINT_OFF

#endif  /* HAVE_SIGSETJMP */

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_INTERRUPT_H_ */
