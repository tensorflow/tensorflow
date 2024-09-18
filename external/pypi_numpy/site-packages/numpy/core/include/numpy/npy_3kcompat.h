/*
 * This is a convenience header file providing compatibility utilities
 * for supporting different minor versions of Python 3.
 * It was originally used to support the transition from Python 2,
 * hence the "3k" naming.
 *
 * If you want to use this for your own projects, it's recommended to make a
 * copy of it. Although the stuff below is unlikely to change, we don't provide
 * strong backwards compatibility guarantees at the moment.
 */

#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_3KCOMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_3KCOMPAT_H_

#include <Python.h>
#include <stdio.h>

#ifndef NPY_PY3K
#define NPY_PY3K 1
#endif

#include "numpy/npy_common.h"
#include "numpy/ndarrayobject.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * PyInt -> PyLong
 */


/*
 * This is a renamed copy of the Python non-limited API function _PyLong_AsInt. It is
 * included here because it is missing from the PyPy API. It completes the PyLong_As*
 * group of functions and can be useful in replacing PyInt_Check.
 */
static NPY_INLINE int
Npy__PyLong_AsInt(PyObject *obj)
{
    int overflow;
    long result = PyLong_AsLongAndOverflow(obj, &overflow);

    /* INT_MAX and INT_MIN are defined in Python.h */
    if (overflow || result > INT_MAX || result < INT_MIN) {
        /* XXX: could be cute and give a different
           message for overflow == -1 */
        PyErr_SetString(PyExc_OverflowError,
                        "Python int too large to convert to C int");
        return -1;
    }
    return (int)result;
}


#if defined(NPY_PY3K)
/* Return True only if the long fits in a C long */
static NPY_INLINE int PyInt_Check(PyObject *op) {
    int overflow = 0;
    if (!PyLong_Check(op)) {
        return 0;
    }
    PyLong_AsLongAndOverflow(op, &overflow);
    return (overflow == 0);
}


#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AS_LONG PyLong_AsLong
#define PyInt_AsSsize_t PyLong_AsSsize_t
#define PyNumber_Int PyNumber_Long

/* NOTE:
 *
 * Since the PyLong type is very different from the fixed-range PyInt,
 * we don't define PyInt_Type -> PyLong_Type.
 */
#endif /* NPY_PY3K */

/* Py3 changes PySlice_GetIndicesEx' first argument's type to PyObject* */
#ifdef NPY_PY3K
#  define NpySlice_GetIndicesEx PySlice_GetIndicesEx
#else
#  define NpySlice_GetIndicesEx(op, nop, start, end, step, slicelength) \
    PySlice_GetIndicesEx((PySliceObject *)op, nop, start, end, step, slicelength)
#endif

#if PY_VERSION_HEX < 0x030900a4
    /* Introduced in https://github.com/python/cpython/commit/d2ec81a8c99796b51fb8c49b77a7fe369863226f */
    #define Py_SET_TYPE(obj, type) ((Py_TYPE(obj) = (type)), (void)0)
    /* Introduced in https://github.com/python/cpython/commit/b10dc3e7a11fcdb97e285882eba6da92594f90f9 */
    #define Py_SET_SIZE(obj, size) ((Py_SIZE(obj) = (size)), (void)0)
    /* Introduced in https://github.com/python/cpython/commit/c86a11221df7e37da389f9c6ce6e47ea22dc44ff */
    #define Py_SET_REFCNT(obj, refcnt) ((Py_REFCNT(obj) = (refcnt)), (void)0)
#endif


#define Npy_EnterRecursiveCall(x) Py_EnterRecursiveCall(x)

/* Py_SETREF was added in 3.5.2, and only if Py_LIMITED_API is absent */
#if PY_VERSION_HEX < 0x03050200
    #define Py_SETREF(op, op2)                      \
        do {                                        \
            PyObject *_py_tmp = (PyObject *)(op);   \
            (op) = (op2);                           \
            Py_DECREF(_py_tmp);                     \
        } while (0)
#endif

/* introduced in https://github.com/python/cpython/commit/a24107b04c1277e3c1105f98aff5bfa3a98b33a0 */
#if PY_VERSION_HEX < 0x030800A3
    static NPY_INLINE PyObject *
    _PyDict_GetItemStringWithError(PyObject *v, const char *key)
    {
        PyObject *kv, *rv;
        kv = PyUnicode_FromString(key);
        if (kv == NULL) {
            return NULL;
        }
        rv = PyDict_GetItemWithError(v, kv);
        Py_DECREF(kv);
        return rv;
    }
#endif

/*
 * PyString -> PyBytes
 */

#if defined(NPY_PY3K)

#define PyString_Type PyBytes_Type
#define PyString_Check PyBytes_Check
#define PyStringObject PyBytesObject
#define PyString_FromString PyBytes_FromString
#define PyString_FromStringAndSize PyBytes_FromStringAndSize
#define PyString_AS_STRING PyBytes_AS_STRING
#define PyString_AsStringAndSize PyBytes_AsStringAndSize
#define PyString_FromFormat PyBytes_FromFormat
#define PyString_Concat PyBytes_Concat
#define PyString_ConcatAndDel PyBytes_ConcatAndDel
#define PyString_AsString PyBytes_AsString
#define PyString_GET_SIZE PyBytes_GET_SIZE
#define PyString_Size PyBytes_Size

#define PyUString_Type PyUnicode_Type
#define PyUString_Check PyUnicode_Check
#define PyUStringObject PyUnicodeObject
#define PyUString_FromString PyUnicode_FromString
#define PyUString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyUString_FromFormat PyUnicode_FromFormat
#define PyUString_Concat PyUnicode_Concat2
#define PyUString_ConcatAndDel PyUnicode_ConcatAndDel
#define PyUString_GET_SIZE PyUnicode_GET_SIZE
#define PyUString_Size PyUnicode_Size
#define PyUString_InternFromString PyUnicode_InternFromString
#define PyUString_Format PyUnicode_Format

#define PyBaseString_Check(obj) (PyUnicode_Check(obj))

#else

#define PyBytes_Type PyString_Type
#define PyBytes_Check PyString_Check
#define PyBytesObject PyStringObject
#define PyBytes_FromString PyString_FromString
#define PyBytes_FromStringAndSize PyString_FromStringAndSize
#define PyBytes_AS_STRING PyString_AS_STRING
#define PyBytes_AsStringAndSize PyString_AsStringAndSize
#define PyBytes_FromFormat PyString_FromFormat
#define PyBytes_Concat PyString_Concat
#define PyBytes_ConcatAndDel PyString_ConcatAndDel
#define PyBytes_AsString PyString_AsString
#define PyBytes_GET_SIZE PyString_GET_SIZE
#define PyBytes_Size PyString_Size

#define PyUString_Type PyString_Type
#define PyUString_Check PyString_Check
#define PyUStringObject PyStringObject
#define PyUString_FromString PyString_FromString
#define PyUString_FromStringAndSize PyString_FromStringAndSize
#define PyUString_FromFormat PyString_FromFormat
#define PyUString_Concat PyString_Concat
#define PyUString_ConcatAndDel PyString_ConcatAndDel
#define PyUString_GET_SIZE PyString_GET_SIZE
#define PyUString_Size PyString_Size
#define PyUString_InternFromString PyString_InternFromString
#define PyUString_Format PyString_Format

#define PyBaseString_Check(obj) (PyBytes_Check(obj) || PyUnicode_Check(obj))

#endif /* NPY_PY3K */


static NPY_INLINE void
PyUnicode_ConcatAndDel(PyObject **left, PyObject *right)
{
    Py_SETREF(*left, PyUnicode_Concat(*left, right));
    Py_DECREF(right);
}

static NPY_INLINE void
PyUnicode_Concat2(PyObject **left, PyObject *right)
{
    Py_SETREF(*left, PyUnicode_Concat(*left, right));
}

/*
 * PyFile_* compatibility
 */

/*
 * Get a FILE* handle to the file represented by the Python object
 */
static NPY_INLINE FILE*
npy_PyFile_Dup2(PyObject *file, char *mode, npy_off_t *orig_pos)
{
    int fd, fd2, unbuf;
    Py_ssize_t fd2_tmp;
    PyObject *ret, *os, *io, *io_raw;
    npy_off_t pos;
    FILE *handle;

    /* For Python 2 PyFileObject, use PyFile_AsFile */
#if !defined(NPY_PY3K)
    if (PyFile_Check(file)) {
        return PyFile_AsFile(file);
    }
#endif

    /* Flush first to ensure things end up in the file in the correct order */
    ret = PyObject_CallMethod(file, "flush", "");
    if (ret == NULL) {
        return NULL;
    }
    Py_DECREF(ret);
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        return NULL;
    }

    /*
     * The handle needs to be dup'd because we have to call fclose
     * at the end
     */
    os = PyImport_ImportModule("os");
    if (os == NULL) {
        return NULL;
    }
    ret = PyObject_CallMethod(os, "dup", "i", fd);
    Py_DECREF(os);
    if (ret == NULL) {
        return NULL;
    }
    fd2_tmp = PyNumber_AsSsize_t(ret, PyExc_IOError);
    Py_DECREF(ret);
    if (fd2_tmp == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (fd2_tmp < INT_MIN || fd2_tmp > INT_MAX) {
        PyErr_SetString(PyExc_IOError,
                        "Getting an 'int' from os.dup() failed");
        return NULL;
    }
    fd2 = (int)fd2_tmp;

    /* Convert to FILE* handle */
#ifdef _WIN32
    handle = _fdopen(fd2, mode);
#else
    handle = fdopen(fd2, mode);
#endif
    if (handle == NULL) {
        PyErr_SetString(PyExc_IOError,
                        "Getting a FILE* from a Python file object failed");
        return NULL;
    }

    /* Record the original raw file handle position */
    *orig_pos = npy_ftell(handle);
    if (*orig_pos == -1) {
        /* The io module is needed to determine if buffering is used */
        io = PyImport_ImportModule("io");
        if (io == NULL) {
            fclose(handle);
            return NULL;
        }
        /* File object instances of RawIOBase are unbuffered */
        io_raw = PyObject_GetAttrString(io, "RawIOBase");
        Py_DECREF(io);
        if (io_raw == NULL) {
            fclose(handle);
            return NULL;
        }
        unbuf = PyObject_IsInstance(file, io_raw);
        Py_DECREF(io_raw);
        if (unbuf == 1) {
            /* Succeed if the IO is unbuffered */
            return handle;
        }
        else {
            PyErr_SetString(PyExc_IOError, "obtaining file position failed");
            fclose(handle);
            return NULL;
        }
    }

    /* Seek raw handle to the Python-side position */
    ret = PyObject_CallMethod(file, "tell", "");
    if (ret == NULL) {
        fclose(handle);
        return NULL;
    }
    pos = PyLong_AsLongLong(ret);
    Py_DECREF(ret);
    if (PyErr_Occurred()) {
        fclose(handle);
        return NULL;
    }
    if (npy_fseek(handle, pos, SEEK_SET) == -1) {
        PyErr_SetString(PyExc_IOError, "seeking file failed");
        fclose(handle);
        return NULL;
    }
    return handle;
}

/*
 * Close the dup-ed file handle, and seek the Python one to the current position
 */
static NPY_INLINE int
npy_PyFile_DupClose2(PyObject *file, FILE* handle, npy_off_t orig_pos)
{
    int fd, unbuf;
    PyObject *ret, *io, *io_raw;
    npy_off_t position;

    /* For Python 2 PyFileObject, do nothing */
#if !defined(NPY_PY3K)
    if (PyFile_Check(file)) {
        return 0;
    }
#endif

    position = npy_ftell(handle);

    /* Close the FILE* handle */
    fclose(handle);

    /*
     * Restore original file handle position, in order to not confuse
     * Python-side data structures
     */
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        return -1;
    }

    if (npy_lseek(fd, orig_pos, SEEK_SET) == -1) {

        /* The io module is needed to determine if buffering is used */
        io = PyImport_ImportModule("io");
        if (io == NULL) {
            return -1;
        }
        /* File object instances of RawIOBase are unbuffered */
        io_raw = PyObject_GetAttrString(io, "RawIOBase");
        Py_DECREF(io);
        if (io_raw == NULL) {
            return -1;
        }
        unbuf = PyObject_IsInstance(file, io_raw);
        Py_DECREF(io_raw);
        if (unbuf == 1) {
            /* Succeed if the IO is unbuffered */
            return 0;
        }
        else {
            PyErr_SetString(PyExc_IOError, "seeking file failed");
            return -1;
        }
    }

    if (position == -1) {
        PyErr_SetString(PyExc_IOError, "obtaining file position failed");
        return -1;
    }

    /* Seek Python-side handle to the FILE* handle position */
    ret = PyObject_CallMethod(file, "seek", NPY_OFF_T_PYFMT "i", position, 0);
    if (ret == NULL) {
        return -1;
    }
    Py_DECREF(ret);
    return 0;
}

static NPY_INLINE int
npy_PyFile_Check(PyObject *file)
{
    int fd;
    /* For Python 2, check if it is a PyFileObject */
#if !defined(NPY_PY3K)
    if (PyFile_Check(file)) {
        return 1;
    }
#endif
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

static NPY_INLINE PyObject*
npy_PyFile_OpenFile(PyObject *filename, const char *mode)
{
    PyObject *open;
    open = PyDict_GetItemString(PyEval_GetBuiltins(), "open");
    if (open == NULL) {
        return NULL;
    }
    return PyObject_CallFunction(open, "Os", filename, mode);
}

static NPY_INLINE int
npy_PyFile_CloseFile(PyObject *file)
{
    PyObject *ret;

    ret = PyObject_CallMethod(file, "close", NULL);
    if (ret == NULL) {
        return -1;
    }
    Py_DECREF(ret);
    return 0;
}


/* This is a copy of _PyErr_ChainExceptions
 */
static NPY_INLINE void
npy_PyErr_ChainExceptions(PyObject *exc, PyObject *val, PyObject *tb)
{
    if (exc == NULL)
        return;

    if (PyErr_Occurred()) {
        /* only py3 supports this anyway */
        #ifdef NPY_PY3K
            PyObject *exc2, *val2, *tb2;
            PyErr_Fetch(&exc2, &val2, &tb2);
            PyErr_NormalizeException(&exc, &val, &tb);
            if (tb != NULL) {
                PyException_SetTraceback(val, tb);
                Py_DECREF(tb);
            }
            Py_DECREF(exc);
            PyErr_NormalizeException(&exc2, &val2, &tb2);
            PyException_SetContext(val2, val);
            PyErr_Restore(exc2, val2, tb2);
        #endif
    }
    else {
        PyErr_Restore(exc, val, tb);
    }
}


/* This is a copy of _PyErr_ChainExceptions, with:
 *  - a minimal implementation for python 2
 *  - __cause__ used instead of __context__
 */
static NPY_INLINE void
npy_PyErr_ChainExceptionsCause(PyObject *exc, PyObject *val, PyObject *tb)
{
    if (exc == NULL)
        return;

    if (PyErr_Occurred()) {
        /* only py3 supports this anyway */
        #ifdef NPY_PY3K
            PyObject *exc2, *val2, *tb2;
            PyErr_Fetch(&exc2, &val2, &tb2);
            PyErr_NormalizeException(&exc, &val, &tb);
            if (tb != NULL) {
                PyException_SetTraceback(val, tb);
                Py_DECREF(tb);
            }
            Py_DECREF(exc);
            PyErr_NormalizeException(&exc2, &val2, &tb2);
            PyException_SetCause(val2, val);
            PyErr_Restore(exc2, val2, tb2);
        #endif
    }
    else {
        PyErr_Restore(exc, val, tb);
    }
}

/*
 * PyObject_Cmp
 */
#if defined(NPY_PY3K)
static NPY_INLINE int
PyObject_Cmp(PyObject *i1, PyObject *i2, int *cmp)
{
    int v;
    v = PyObject_RichCompareBool(i1, i2, Py_LT);
    if (v == 1) {
        *cmp = -1;
        return 1;
    }
    else if (v == -1) {
        return -1;
    }

    v = PyObject_RichCompareBool(i1, i2, Py_GT);
    if (v == 1) {
        *cmp = 1;
        return 1;
    }
    else if (v == -1) {
        return -1;
    }

    v = PyObject_RichCompareBool(i1, i2, Py_EQ);
    if (v == 1) {
        *cmp = 0;
        return 1;
    }
    else {
        *cmp = 0;
        return -1;
    }
}
#endif

/*
 * PyCObject functions adapted to PyCapsules.
 *
 * The main job here is to get rid of the improved error handling
 * of PyCapsules. It's a shame...
 */
static NPY_INLINE PyObject *
NpyCapsule_FromVoidPtr(void *ptr, void (*dtor)(PyObject *))
{
    PyObject *ret = PyCapsule_New(ptr, NULL, dtor);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

static NPY_INLINE PyObject *
NpyCapsule_FromVoidPtrAndDesc(void *ptr, void* context, void (*dtor)(PyObject *))
{
    PyObject *ret = NpyCapsule_FromVoidPtr(ptr, dtor);
    if (ret != NULL && PyCapsule_SetContext(ret, context) != 0) {
        PyErr_Clear();
        Py_DECREF(ret);
        ret = NULL;
    }
    return ret;
}

static NPY_INLINE void *
NpyCapsule_AsVoidPtr(PyObject *obj)
{
    void *ret = PyCapsule_GetPointer(obj, NULL);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

static NPY_INLINE void *
NpyCapsule_GetDesc(PyObject *obj)
{
    return PyCapsule_GetContext(obj);
}

static NPY_INLINE int
NpyCapsule_Check(PyObject *ptr)
{
    return PyCapsule_CheckExact(ptr);
}

#ifdef __cplusplus
}
#endif


#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_3KCOMPAT_H_ */
