/*
 * This header exports the new experimental DType API as proposed in
 * NEPs 41 to 43.  For background, please check these NEPs.  Otherwise,
 * this header also serves as documentation for the time being.
 *
 * Please do not hesitate to contact @seberg with questions.  This is
 * developed together with https://github.com/seberg/experimental_user_dtypes
 * and those interested in experimenting are encouraged to contribute there.
 *
 * To use the functions defined in the header, call::
 *
 *     if (import_experimental_dtype_api(version) < 0) {
 *         return NULL;
 *     }
 *
 * in your module init.  (A version mismatch will be reported, just update
 * to the correct one, this will alert you of possible changes.)
 *
 * The following lists the main symbols currently exported.  Please do not
 * hesitate to ask for help or clarification:
 *
 * - PyUFunc_AddLoopFromSpec:
 *
 *     Register a new loop for a ufunc.  This uses the `PyArrayMethod_Spec`
 *     which must be filled in (see in-line comments).
 *
 * - PyUFunc_AddWrappingLoop:
 *
 *     Register a new loop which reuses an existing one, but modifies the
 *     result dtypes.  Please search the internal NumPy docs for more info
 *     at this point.  (Used for physical units dtype.)
 *
 * - PyUFunc_AddPromoter:
 *
 *     Register a new promoter for a ufunc.  A promoter is a function stored
 *     in a PyCapsule (see in-line comments).  It is passed the operation and
 *     requested DType signatures and can mutate it to attempt a new search
 *     for a matching loop/promoter.
 *     I.e. for Numba a promoter could even add the desired loop.
 *
 * - PyArrayInitDTypeMeta_FromSpec:
 *
 *     Initialize a new DType.  It must currently be a static Python C type
 *     that is declared as `PyArray_DTypeMeta` and not `PyTypeObject`.
 *     Further, it must subclass `np.dtype` and set its type to
 *     `PyArrayDTypeMeta_Type` (before calling `PyType_Read()`).
 *
 * - PyArray_CommonDType:
 *
 *     Find the common-dtype ("promotion") for two DType classes.  Similar
 *     to `np.result_type`, but works on the classes and not instances.
 *
 * - PyArray_PromoteDTypeSequence:
 *
 *     Same as CommonDType, but works with an arbitrary number of DTypes.
 *     This function is smarter and can often return successful and unambiguous
 *     results when `common_dtype(common_dtype(dt1, dt2), dt3)` would
 *     depend on the operation order or fail.  Nevertheless, DTypes should
 *     aim to ensure that their common-dtype implementation is associative
 *     and commutative!  (Mainly, unsigned and signed integers are not.)
 *
 *     For guaranteed consistent results DTypes must implement common-Dtype
 *     "transitively".  If A promotes B and B promotes C, than A must generally
 *     also promote C; where "promotes" means implements the promotion.
 *     (There are some exceptions for abstract DTypes)
 *
 * - PyArray_GetDefaultDescr:
 *
 *     Given a DType class, returns the default instance (descriptor).
 *     This is an inline function checking for `singleton` first and only
 *     calls the `default_descr` function if necessary.
 *
 * - PyArray_DoubleDType, etc.:
 *
 *     Aliases to the DType classes for the builtin NumPy DTypes.
 *
 * WARNING
 * =======
 *
 * By using this header, you understand that this is a fully experimental
 * exposure.  Details are expected to change, and some options may have no
 * effect.  (Please contact @seberg if you have questions!)
 * If the exposure stops working, please file a bug report with NumPy.
 * Further, a DType created using this API/header should still be expected
 * to be incompatible with some functionality inside and outside of NumPy.
 * In this case crashes must be expected.  Please report any such problems
 * so that they can be fixed before final exposure.
 * Furthermore, expect missing checks for programming errors which the final
 * API is expected to have.
 *
 * Symbols with a leading underscore are likely to not be included in the
 * first public version, if these are central to your use-case, please let
 * us know, so that we can reconsider.
 *
 * "Array-like" consumer API not yet under considerations
 * ======================================================
 *
 * The new DType API is designed in a way to make it potentially useful for
 * alternative "array-like" implementations.  This will require careful
 * exposure of details and functions and is not part of this experimental API.
 *
 * Brief (incompatibility) changelog
 * =================================
 *
 * 2. None (only additions).
 * 3. New `npy_intp *view_offset` argument for `resolve_descriptors`.
 *    This replaces the `NPY_CAST_IS_VIEW` flag.  It can be set to 0 if the
 *    operation is a view, and is pre-initialized to `NPY_MIN_INTP` indicating
 *    that the operation is not a view.
 */

#ifndef NUMPY_CORE_INCLUDE_NUMPY_EXPERIMENTAL_DTYPE_API_H_
#define NUMPY_CORE_INCLUDE_NUMPY_EXPERIMENTAL_DTYPE_API_H_

#include <Python.h>
#include "ndarraytypes.h"


/*
 * There must be a better way?! -- Oh well, this is experimental
 * (my issue with it, is that I cannot undef those helpers).
 */
#if defined(PY_ARRAY_UNIQUE_SYMBOL)
    #define NPY_EXP_DTYPE_API_CONCAT_HELPER2(x, y) x ## y
    #define NPY_EXP_DTYPE_API_CONCAT_HELPER(arg) NPY_EXP_DTYPE_API_CONCAT_HELPER2(arg, __experimental_dtype_api_table)
    #define __experimental_dtype_api_table NPY_EXP_DTYPE_API_CONCAT_HELPER(PY_ARRAY_UNIQUE_SYMBOL)
#else
    #define __experimental_dtype_api_table __experimental_dtype_api_table
#endif

/* Support for correct multi-file projects: */
#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
    extern void **__experimental_dtype_api_table;
#else
    /*
     * Just a hack so I don't forget importing as much myself, I spend way too
     * much time noticing it the first time around :).
     */
    static void
    __not_imported(void)
    {
        printf("*****\nCritical error, dtype API not imported\n*****\n");
    }

    static void *__uninitialized_table[] = {
            &__not_imported, &__not_imported, &__not_imported, &__not_imported,
            &__not_imported, &__not_imported, &__not_imported, &__not_imported};

    #if defined(PY_ARRAY_UNIQUE_SYMBOL)
        void **__experimental_dtype_api_table = __uninitialized_table;
    #else
        static void **__experimental_dtype_api_table = __uninitialized_table;
    #endif
#endif


/*
 * DTypeMeta struct, the content may be made fully opaque (except the size).
 * We may also move everything into a single `void *dt_slots`.
 */
typedef struct {
    PyHeapTypeObject super;
    PyArray_Descr *singleton;
    int type_num;
    PyTypeObject *scalar_type;
    npy_uint64 flags;
    void *dt_slots;
    void *reserved[3];
} PyArray_DTypeMeta;


/*
 * ******************************************************
 *         ArrayMethod API (Casting and UFuncs)
 * ******************************************************
 */
/*
 * NOTE: Expected changes:
 *       * invert logic of floating point error flag
 *       * probably split runtime and general flags into two
 *       * should possibly not use an enum for typedef for more stable ABI?
 */
typedef enum {
    /* Flag for whether the GIL is required */
    NPY_METH_REQUIRES_PYAPI = 1 << 1,
    /*
     * Some functions cannot set floating point error flags, this flag
     * gives us the option (not requirement) to skip floating point error
     * setup/check. No function should set error flags and ignore them
     * since it would interfere with chaining operations (e.g. casting).
     */
    NPY_METH_NO_FLOATINGPOINT_ERRORS = 1 << 2,
    /* Whether the method supports unaligned access (not runtime) */
    NPY_METH_SUPPORTS_UNALIGNED = 1 << 3,

    /* All flags which can change at runtime */
    NPY_METH_RUNTIME_FLAGS = (
            NPY_METH_REQUIRES_PYAPI |
            NPY_METH_NO_FLOATINGPOINT_ERRORS),
} NPY_ARRAYMETHOD_FLAGS;


/*
 * The main object for creating a new ArrayMethod. We use the typical `slots`
 * mechanism used by the Python limited API (see below for the slot defs).
 */
typedef struct {
    const char *name;
    int nin, nout;
    NPY_CASTING casting;
    NPY_ARRAYMETHOD_FLAGS flags;
    PyArray_DTypeMeta **dtypes;
    PyType_Slot *slots;
} PyArrayMethod_Spec;


typedef PyObject *_ufunc_addloop_fromspec_func(
        PyObject *ufunc, PyArrayMethod_Spec *spec);
/*
 * The main ufunc registration function.  This adds a new implementation/loop
 * to a ufunc.  It replaces `PyUFunc_RegisterLoopForType`.
 */
#define PyUFunc_AddLoopFromSpec \
    (*(_ufunc_addloop_fromspec_func *)(__experimental_dtype_api_table[0]))


/* Please see the NumPy definitions in `array_method.h` for details on these */
typedef int translate_given_descrs_func(int nin, int nout,
        PyArray_DTypeMeta *wrapped_dtypes[],
        PyArray_Descr *given_descrs[], PyArray_Descr *new_descrs[]);
typedef int translate_loop_descrs_func(int nin, int nout,
        PyArray_DTypeMeta *new_dtypes[], PyArray_Descr *given_descrs[],
        PyArray_Descr *original_descrs[], PyArray_Descr *loop_descrs[]);

typedef int _ufunc_wrapping_loop_func(PyObject *ufunc_obj,
        PyArray_DTypeMeta *new_dtypes[], PyArray_DTypeMeta *wrapped_dtypes[],
        translate_given_descrs_func *translate_given_descrs,
        translate_loop_descrs_func *translate_loop_descrs);
#define PyUFunc_AddWrappingLoop \
    (*(_ufunc_wrapping_loop_func *)(__experimental_dtype_api_table[7]))

/*
 * Type of the C promoter function, which must be wrapped into a
 * PyCapsule with name "numpy._ufunc_promoter".
 *
 * Note that currently the output dtypes are always NULL unless they are
 * also part of the signature.  This is an implementation detail and could
 * change in the future.  However, in general promoters should not have a
 * need for output dtypes.
 * (There are potential use-cases, these are currently unsupported.)
 */
typedef int promoter_function(PyObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[]);

/*
 * Function to register a promoter.
 *
 * @param ufunc The ufunc object to register the promoter with.
 * @param DType_tuple A Python tuple containing DTypes or None matching the
 *        number of inputs and outputs of the ufunc.
 * @param promoter A PyCapsule with name "numpy._ufunc_promoter" containing
 *        a pointer to a `promoter_function`.
 */
typedef int _ufunc_addpromoter_func(
        PyObject *ufunc, PyObject *DType_tuple, PyObject *promoter);
#define PyUFunc_AddPromoter \
    (*(_ufunc_addpromoter_func *)(__experimental_dtype_api_table[1]))


/*
 * The resolve descriptors function, must be able to handle NULL values for
 * all output (but not input) `given_descrs` and fill `loop_descrs`.
 * Return -1 on error or 0 if the operation is not possible without an error
 * set.  (This may still be in flux.)
 * Otherwise must return the "casting safety", for normal functions, this is
 * almost always "safe" (or even "equivalent"?).
 *
 * `resolve_descriptors` is optional if all output DTypes are non-parametric.
 */
#define NPY_METH_resolve_descriptors 1
typedef NPY_CASTING (resolve_descriptors_function)(
        /* "method" is currently opaque (necessary e.g. to wrap Python) */
        PyObject *method,
        /* DTypes the method was created for */
        PyObject **dtypes,
        /* Input descriptors (instances).  Outputs may be NULL. */
        PyArray_Descr **given_descrs,
        /* Exact loop descriptors to use, must not hold references on error */
        PyArray_Descr **loop_descrs,
        npy_intp *view_offset);

/* NOT public yet: Signature needs adapting as external API. */
#define _NPY_METH_get_loop 2

/*
 * Current public API to define fast inner-loops.  You must provide a
 * strided loop.  If this is a cast between two "versions" of the same dtype
 * you must also provide an unaligned strided loop.
 * Other loops are useful to optimize the very common contiguous case.
 *
 * NOTE: As of now, NumPy will NOT use unaligned loops in ufuncs!
 */
#define NPY_METH_strided_loop 3
#define NPY_METH_contiguous_loop 4
#define NPY_METH_unaligned_strided_loop 5
#define NPY_METH_unaligned_contiguous_loop 6


typedef struct {
    PyObject *caller;  /* E.g. the original ufunc, may be NULL */
    PyObject *method;  /* The method "self". Currently an opaque object */

    /* Operand descriptors, filled in by resolve_descriptors */
    PyArray_Descr **descriptors;
    /* Structure may grow (this is harmless for DType authors) */
} PyArrayMethod_Context;

typedef int (PyArrayMethod_StridedLoop)(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *transferdata);



/*
 * ****************************
 *          DTYPE API
 * ****************************
 */

#define NPY_DT_ABSTRACT 1 << 1
#define NPY_DT_PARAMETRIC 1 << 2

#define NPY_DT_discover_descr_from_pyobject 1
#define _NPY_DT_is_known_scalar_type 2
#define NPY_DT_default_descr 3
#define NPY_DT_common_dtype 4
#define NPY_DT_common_instance 5
#define NPY_DT_setitem 6
#define NPY_DT_getitem 7


// TODO: These slots probably still need some thought, and/or a way to "grow"?
typedef struct{
    PyTypeObject *typeobj;    /* type of python scalar or NULL */
    int flags;                /* flags, including parametric and abstract */
    /* NULL terminated cast definitions. Use NULL for the newly created DType */
    PyArrayMethod_Spec **casts;
    PyType_Slot *slots;
    /* Baseclass or NULL (will always subclass `np.dtype`) */
    PyTypeObject *baseclass;
} PyArrayDTypeMeta_Spec;


#define PyArrayDTypeMeta_Type \
    (*(PyTypeObject *)__experimental_dtype_api_table[2])
typedef int __dtypemeta_fromspec(
        PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *dtype_spec);
/*
 * Finalize creation of a DTypeMeta.  You must ensure that the DTypeMeta is
 * a proper subclass.  The DTypeMeta object has additional fields compared to
 * a normal PyTypeObject!
 * The only (easy) creation of a new DType is to create a static Type which
 * inherits `PyArray_DescrType`, sets its type to `PyArrayDTypeMeta_Type` and
 * uses `PyArray_DTypeMeta` defined above as the C-structure.
 */
#define PyArrayInitDTypeMeta_FromSpec \
    ((__dtypemeta_fromspec *)(__experimental_dtype_api_table[3]))


/*
 * *************************************
 *          WORKING WITH DTYPES
 * *************************************
 */

typedef PyArray_DTypeMeta *__common_dtype(
        PyArray_DTypeMeta *DType1, PyArray_DTypeMeta *DType2);
#define PyArray_CommonDType \
    ((__common_dtype *)(__experimental_dtype_api_table[4]))


typedef PyArray_DTypeMeta *__promote_dtype_sequence(
        npy_intp num, PyArray_DTypeMeta *DTypes[]);
#define PyArray_PromoteDTypeSequence \
    ((__promote_dtype_sequence *)(__experimental_dtype_api_table[5]))


typedef PyArray_Descr *__get_default_descr(
        PyArray_DTypeMeta *DType);
#define _PyArray_GetDefaultDescr \
    ((__get_default_descr *)(__experimental_dtype_api_table[6]))

static NPY_INLINE PyArray_Descr *
PyArray_GetDefaultDescr(PyArray_DTypeMeta *DType)
{
    if (DType->singleton != NULL) {
        Py_INCREF(DType->singleton);
        return DType->singleton;
    }
    return _PyArray_GetDefaultDescr(DType);
}


/*
 * NumPy's builtin DTypes:
 */
#define PyArray_BoolDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[10])
/* Integers */
#define PyArray_ByteDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[11])
#define PyArray_UByteDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[12])
#define PyArray_ShortDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[13])
#define PyArray_UShortDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[14])
#define PyArray_IntDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[15])
#define PyArray_UIntDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[16])
#define PyArray_LongDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[17])
#define PyArray_ULongDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[18])
#define PyArray_LongLongDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[19])
#define PyArray_ULongLongDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[20])
/* Integer aliases */
#define PyArray_Int8Type (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[21])
#define PyArray_UInt8DType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[22])
#define PyArray_Int16DType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[23])
#define PyArray_UInt16DType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[24])
#define PyArray_Int32DType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[25])
#define PyArray_UInt32DType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[26])
#define PyArray_Int64DType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[27])
#define PyArray_UInt64DType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[28])
#define PyArray_IntpDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[29])
#define PyArray_UIntpDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[30])
/* Floats */
#define PyArray_HalfType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[31])
#define PyArray_FloatDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[32])
#define PyArray_DoubleDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[33])
#define PyArray_LongDoubleDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[34])
/* Complex */
#define PyArray_CFloatDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[35])
#define PyArray_CDoubleDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[36])
#define PyArray_CLongDoubleDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[37])
/* String/Bytes */
#define PyArray_StringDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[38])
#define PyArray_UnicodeDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[39])
/* Datetime/Timedelta */
#define PyArray_DatetimeDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[40])
#define PyArray_TimedeltaDType (*(PyArray_DTypeMeta *)__experimental_dtype_api_table[41])


/*
 * ********************************
 *         Initialization
 * ********************************
 *
 * Import the experimental API, the version must match the one defined in
 * the header to ensure changes are taken into account. NumPy will further
 * runtime-check this.
 * You must call this function to use the symbols defined in this file.
 */
#if !defined(NO_IMPORT) && !defined(NO_IMPORT_ARRAY)

#define __EXPERIMENTAL_DTYPE_VERSION 4

static int
import_experimental_dtype_api(int version)
{
    if (version != __EXPERIMENTAL_DTYPE_VERSION) {
        PyErr_Format(PyExc_RuntimeError,
                "DType API version %d did not match header version %d. Please "
                "update the import statement and check for API changes.",
                version, __EXPERIMENTAL_DTYPE_VERSION);
        return -1;
    }
    if (__experimental_dtype_api_table != __uninitialized_table) {
        /* already imported. */
        return 0;
    }

    PyObject *multiarray = PyImport_ImportModule("numpy.core._multiarray_umath");
    if (multiarray == NULL) {
        return -1;
    }

    PyObject *api = PyObject_CallMethod(multiarray,
        "_get_experimental_dtype_api", "i", version);
    Py_DECREF(multiarray);
    if (api == NULL) {
        return -1;
    }
    __experimental_dtype_api_table = (void **)PyCapsule_GetPointer(api,
            "experimental_dtype_api_table");
    Py_DECREF(api);

    if (__experimental_dtype_api_table == NULL) {
        __experimental_dtype_api_table = __uninitialized_table;
        return -1;
    }
    return 0;
}

#endif  /* !defined(NO_IMPORT) && !defined(NO_IMPORT_ARRAY) */

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_EXPERIMENTAL_DTYPE_API_H_ */
