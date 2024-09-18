#ifndef NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_
#define NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_

#ifndef _MULTIARRAYMODULE
typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;
#endif


typedef struct {
        PyObject_HEAD
        signed char obval;
} PyByteScalarObject;


typedef struct {
        PyObject_HEAD
        short obval;
} PyShortScalarObject;


typedef struct {
        PyObject_HEAD
        int obval;
} PyIntScalarObject;


typedef struct {
        PyObject_HEAD
        long obval;
} PyLongScalarObject;


typedef struct {
        PyObject_HEAD
        npy_longlong obval;
} PyLongLongScalarObject;


typedef struct {
        PyObject_HEAD
        unsigned char obval;
} PyUByteScalarObject;


typedef struct {
        PyObject_HEAD
        unsigned short obval;
} PyUShortScalarObject;


typedef struct {
        PyObject_HEAD
        unsigned int obval;
} PyUIntScalarObject;


typedef struct {
        PyObject_HEAD
        unsigned long obval;
} PyULongScalarObject;


typedef struct {
        PyObject_HEAD
        npy_ulonglong obval;
} PyULongLongScalarObject;


typedef struct {
        PyObject_HEAD
        npy_half obval;
} PyHalfScalarObject;


typedef struct {
        PyObject_HEAD
        float obval;
} PyFloatScalarObject;


typedef struct {
        PyObject_HEAD
        double obval;
} PyDoubleScalarObject;


typedef struct {
        PyObject_HEAD
        npy_longdouble obval;
} PyLongDoubleScalarObject;


typedef struct {
        PyObject_HEAD
        npy_cfloat obval;
} PyCFloatScalarObject;


typedef struct {
        PyObject_HEAD
        npy_cdouble obval;
} PyCDoubleScalarObject;


typedef struct {
        PyObject_HEAD
        npy_clongdouble obval;
} PyCLongDoubleScalarObject;


typedef struct {
        PyObject_HEAD
        PyObject * obval;
} PyObjectScalarObject;

typedef struct {
        PyObject_HEAD
        npy_datetime obval;
        PyArray_DatetimeMetaData obmeta;
} PyDatetimeScalarObject;

typedef struct {
        PyObject_HEAD
        npy_timedelta obval;
        PyArray_DatetimeMetaData obmeta;
} PyTimedeltaScalarObject;


typedef struct {
        PyObject_HEAD
        char obval;
} PyScalarObject;

#define PyStringScalarObject PyBytesObject
typedef struct {
        /* note that the PyObject_HEAD macro lives right here */
        PyUnicodeObject base;
        Py_UCS4 *obval;
        char *buffer_fmt;
} PyUnicodeScalarObject;


typedef struct {
        PyObject_VAR_HEAD
        char *obval;
        PyArray_Descr *descr;
        int flags;
        PyObject *base;
        void *_buffer_info;  /* private buffer info, tagged to allow warning */
} PyVoidScalarObject;

/* Macros
     Py<Cls><bitsize>ScalarObject
     Py<Cls><bitsize>ArrType_Type
   are defined in ndarrayobject.h
*/

#define PyArrayScalar_False ((PyObject *)(&(_PyArrayScalar_BoolValues[0])))
#define PyArrayScalar_True ((PyObject *)(&(_PyArrayScalar_BoolValues[1])))
#define PyArrayScalar_FromLong(i) \
        ((PyObject *)(&(_PyArrayScalar_BoolValues[((i)!=0)])))
#define PyArrayScalar_RETURN_BOOL_FROM_LONG(i)                  \
        return Py_INCREF(PyArrayScalar_FromLong(i)), \
                PyArrayScalar_FromLong(i)
#define PyArrayScalar_RETURN_FALSE              \
        return Py_INCREF(PyArrayScalar_False),  \
                PyArrayScalar_False
#define PyArrayScalar_RETURN_TRUE               \
        return Py_INCREF(PyArrayScalar_True),   \
                PyArrayScalar_True

#define PyArrayScalar_New(cls) \
        Py##cls##ArrType_Type.tp_alloc(&Py##cls##ArrType_Type, 0)
#define PyArrayScalar_VAL(obj, cls)             \
        ((Py##cls##ScalarObject *)obj)->obval
#define PyArrayScalar_ASSIGN(obj, cls, val) \
        PyArrayScalar_VAL(obj, cls) = val

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_ARRAYSCALARS_H_ */
