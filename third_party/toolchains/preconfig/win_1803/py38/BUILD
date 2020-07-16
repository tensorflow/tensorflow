licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# Point both runtimes to the same python binary to ensure we always
# use the python binary specified by ./configure.py script.
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

py_runtime(
    name = "py2_runtime",
    interpreter_path = "c:/Python38/python.exe",
    python_version = "PY2",
)

py_runtime(
    name = "py3_runtime",
    interpreter_path = "c:/Python38/python.exe",
    python_version = "PY3",
)

py_runtime_pair(
    name = "py_runtime_pair",
    py2_runtime = ":py2_runtime",
    py3_runtime = ":py3_runtime",
)

toolchain(
    name = "py_toolchain",
    toolchain = ":py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)

# To build Python C/C++ extension on Windows, we need to link to python import library pythonXY.lib
# See https://docs.python.org/3/extending/windows.html
cc_import(
    name = "python_lib",
    interface_library = select({
        ":windows": ":python_import_lib",
        # A placeholder for Unix platforms which makes --no_build happy.
        "//conditions:default": "not-existing.lib",
    }),
    system_provided = 1,
)

cc_library(
    name = "python_headers",
    hdrs = [":python_include"],
    includes = ["python_include"],
    deps = select({
        ":windows": [":python_lib"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "numpy_headers",
    hdrs = [":numpy_include"],
    includes = ["numpy_include"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

genrule(
    name = "python_include",
    outs = [
        "python_include/Python-ast.h",
        "python_include/Python.h",
        "python_include/abstract.h",
        "python_include/asdl.h",
        "python_include/ast.h",
        "python_include/bitset.h",
        "python_include/bltinmodule.h",
        "python_include/boolobject.h",
        "python_include/bytearrayobject.h",
        "python_include/bytes_methods.h",
        "python_include/bytesobject.h",
        "python_include/cellobject.h",
        "python_include/ceval.h",
        "python_include/classobject.h",
        "python_include/code.h",
        "python_include/codecs.h",
        "python_include/compile.h",
        "python_include/complexobject.h",
        "python_include/context.h",
        "python_include/cpython/abstract.h",
        "python_include/cpython/dictobject.h",
        "python_include/cpython/fileobject.h",
        "python_include/cpython/initconfig.h",
        "python_include/cpython/interpreteridobject.h",
        "python_include/cpython/object.h",
        "python_include/cpython/objimpl.h",
        "python_include/cpython/pyerrors.h",
        "python_include/cpython/pylifecycle.h",
        "python_include/cpython/pymem.h",
        "python_include/cpython/pystate.h",
        "python_include/cpython/sysmodule.h",
        "python_include/cpython/traceback.h",
        "python_include/cpython/tupleobject.h",
        "python_include/cpython/unicodeobject.h",
        "python_include/datetime.h",
        "python_include/descrobject.h",
        "python_include/dictobject.h",
        "python_include/dtoa.h",
        "python_include/dynamic_annotations.h",
        "python_include/enumobject.h",
        "python_include/errcode.h",
        "python_include/eval.h",
        "python_include/fileobject.h",
        "python_include/fileutils.h",
        "python_include/floatobject.h",
        "python_include/frameobject.h",
        "python_include/funcobject.h",
        "python_include/genobject.h",
        "python_include/graminit.h",
        "python_include/grammar.h",
        "python_include/import.h",
        "python_include/internal/pycore_accu.h",
        "python_include/internal/pycore_atomic.h",
        "python_include/internal/pycore_ceval.h",
        "python_include/internal/pycore_code.h",
        "python_include/internal/pycore_condvar.h",
        "python_include/internal/pycore_context.h",
        "python_include/internal/pycore_fileutils.h",
        "python_include/internal/pycore_getopt.h",
        "python_include/internal/pycore_gil.h",
        "python_include/internal/pycore_hamt.h",
        "python_include/internal/pycore_initconfig.h",
        "python_include/internal/pycore_object.h",
        "python_include/internal/pycore_pathconfig.h",
        "python_include/internal/pycore_pyerrors.h",
        "python_include/internal/pycore_pyhash.h",
        "python_include/internal/pycore_pylifecycle.h",
        "python_include/internal/pycore_pymem.h",
        "python_include/internal/pycore_pystate.h",
        "python_include/internal/pycore_traceback.h",
        "python_include/internal/pycore_tupleobject.h",
        "python_include/internal/pycore_warnings.h",
        "python_include/interpreteridobject.h",
        "python_include/intrcheck.h",
        "python_include/iterobject.h",
        "python_include/listobject.h",
        "python_include/longintrepr.h",
        "python_include/longobject.h",
        "python_include/marshal.h",
        "python_include/memoryobject.h",
        "python_include/methodobject.h",
        "python_include/modsupport.h",
        "python_include/moduleobject.h",
        "python_include/namespaceobject.h",
        "python_include/node.h",
        "python_include/object.h",
        "python_include/objimpl.h",
        "python_include/odictobject.h",
        "python_include/opcode.h",
        "python_include/osdefs.h",
        "python_include/osmodule.h",
        "python_include/parsetok.h",
        "python_include/patchlevel.h",
        "python_include/picklebufobject.h",
        "python_include/py_curses.h",
        "python_include/pyarena.h",
        "python_include/pycapsule.h",
        "python_include/pyconfig.h",
        "python_include/pyctype.h",
        "python_include/pydebug.h",
        "python_include/pydtrace.h",
        "python_include/pyerrors.h",
        "python_include/pyexpat.h",
        "python_include/pyfpe.h",
        "python_include/pyhash.h",
        "python_include/pylifecycle.h",
        "python_include/pymacconfig.h",
        "python_include/pymacro.h",
        "python_include/pymath.h",
        "python_include/pymem.h",
        "python_include/pyport.h",
        "python_include/pystate.h",
        "python_include/pystrcmp.h",
        "python_include/pystrhex.h",
        "python_include/pystrtod.h",
        "python_include/pythonrun.h",
        "python_include/pythread.h",
        "python_include/pytime.h",
        "python_include/rangeobject.h",
        "python_include/setobject.h",
        "python_include/sliceobject.h",
        "python_include/structmember.h",
        "python_include/structseq.h",
        "python_include/symtable.h",
        "python_include/sysmodule.h",
        "python_include/token.h",
        "python_include/traceback.h",
        "python_include/tracemalloc.h",
        "python_include/tupleobject.h",
        "python_include/typeslots.h",
        "python_include/ucnhash.h",
        "python_include/unicodeobject.h",
        "python_include/warnings.h",
        "python_include/weakrefobject.h",
    ],
    cmd = """
cp -f "c:/Python38/include/Python-ast.h" "$(@D)/python_include/Python-ast.h" && cp -f "c:/Python38/include/Python.h" "$(@D)/python_include/Python.h" && cp -f "c:/Python38/include/abstract.h" "$(@D)/python_include/abstract.h" && cp -f "c:/Python38/include/asdl.h" "$(@D)/python_include/asdl.h" && cp -f "c:/Python38/include/ast.h" "$(@D)/python_include/ast.h" && cp -f "c:/Python38/include/bitset.h" "$(@D)/python_include/bitset.h" && cp -f "c:/Python38/include/bltinmodule.h" "$(@D)/python_include/bltinmodule.h" && cp -f "c:/Python38/include/boolobject.h" "$(@D)/python_include/boolobject.h" && cp -f "c:/Python38/include/bytearrayobject.h" "$(@D)/python_include/bytearrayobject.h" && cp -f "c:/Python38/include/bytes_methods.h" "$(@D)/python_include/bytes_methods.h" && cp -f "c:/Python38/include/bytesobject.h" "$(@D)/python_include/bytesobject.h" && cp -f "c:/Python38/include/cellobject.h" "$(@D)/python_include/cellobject.h" && cp -f "c:/Python38/include/ceval.h" "$(@D)/python_include/ceval.h" && cp -f "c:/Python38/include/classobject.h" "$(@D)/python_include/classobject.h" && cp -f "c:/Python38/include/code.h" "$(@D)/python_include/code.h" && cp -f "c:/Python38/include/codecs.h" "$(@D)/python_include/codecs.h" && cp -f "c:/Python38/include/compile.h" "$(@D)/python_include/compile.h" && cp -f "c:/Python38/include/complexobject.h" "$(@D)/python_include/complexobject.h" && cp -f "c:/Python38/include/context.h" "$(@D)/python_include/context.h" && cp -f "c:/Python38/include/cpython/abstract.h" "$(@D)/python_include/cpython/abstract.h" && cp -f "c:/Python38/include/cpython/dictobject.h" "$(@D)/python_include/cpython/dictobject.h" && cp -f "c:/Python38/include/cpython/fileobject.h" "$(@D)/python_include/cpython/fileobject.h" && cp -f "c:/Python38/include/cpython/initconfig.h" "$(@D)/python_include/cpython/initconfig.h" && cp -f "c:/Python38/include/cpython/interpreteridobject.h" "$(@D)/python_include/cpython/interpreteridobject.h" && cp -f "c:/Python38/include/cpython/object.h" "$(@D)/python_include/cpython/object.h" && cp -f "c:/Python38/include/cpython/objimpl.h" "$(@D)/python_include/cpython/objimpl.h" && cp -f "c:/Python38/include/cpython/pyerrors.h" "$(@D)/python_include/cpython/pyerrors.h" && cp -f "c:/Python38/include/cpython/pylifecycle.h" "$(@D)/python_include/cpython/pylifecycle.h" && cp -f "c:/Python38/include/cpython/pymem.h" "$(@D)/python_include/cpython/pymem.h" && cp -f "c:/Python38/include/cpython/pystate.h" "$(@D)/python_include/cpython/pystate.h" && cp -f "c:/Python38/include/cpython/sysmodule.h" "$(@D)/python_include/cpython/sysmodule.h" && cp -f "c:/Python38/include/cpython/traceback.h" "$(@D)/python_include/cpython/traceback.h" && cp -f "c:/Python38/include/cpython/tupleobject.h" "$(@D)/python_include/cpython/tupleobject.h" && cp -f "c:/Python38/include/cpython/unicodeobject.h" "$(@D)/python_include/cpython/unicodeobject.h" && cp -f "c:/Python38/include/datetime.h" "$(@D)/python_include/datetime.h" && cp -f "c:/Python38/include/descrobject.h" "$(@D)/python_include/descrobject.h" && cp -f "c:/Python38/include/dictobject.h" "$(@D)/python_include/dictobject.h" && cp -f "c:/Python38/include/dtoa.h" "$(@D)/python_include/dtoa.h" && cp -f "c:/Python38/include/dynamic_annotations.h" "$(@D)/python_include/dynamic_annotations.h" && cp -f "c:/Python38/include/enumobject.h" "$(@D)/python_include/enumobject.h" && cp -f "c:/Python38/include/errcode.h" "$(@D)/python_include/errcode.h" && cp -f "c:/Python38/include/eval.h" "$(@D)/python_include/eval.h" && cp -f "c:/Python38/include/fileobject.h" "$(@D)/python_include/fileobject.h" && cp -f "c:/Python38/include/fileutils.h" "$(@D)/python_include/fileutils.h" && cp -f "c:/Python38/include/floatobject.h" "$(@D)/python_include/floatobject.h" && cp -f "c:/Python38/include/frameobject.h" "$(@D)/python_include/frameobject.h" && cp -f "c:/Python38/include/funcobject.h" "$(@D)/python_include/funcobject.h" && cp -f "c:/Python38/include/genobject.h" "$(@D)/python_include/genobject.h" && cp -f "c:/Python38/include/graminit.h" "$(@D)/python_include/graminit.h" && cp -f "c:/Python38/include/grammar.h" "$(@D)/python_include/grammar.h" && cp -f "c:/Python38/include/import.h" "$(@D)/python_include/import.h" && cp -f "c:/Python38/include/internal/pycore_accu.h" "$(@D)/python_include/internal/pycore_accu.h" && cp -f "c:/Python38/include/internal/pycore_atomic.h" "$(@D)/python_include/internal/pycore_atomic.h" && cp -f "c:/Python38/include/internal/pycore_ceval.h" "$(@D)/python_include/internal/pycore_ceval.h" && cp -f "c:/Python38/include/internal/pycore_code.h" "$(@D)/python_include/internal/pycore_code.h" && cp -f "c:/Python38/include/internal/pycore_condvar.h" "$(@D)/python_include/internal/pycore_condvar.h" && cp -f "c:/Python38/include/internal/pycore_context.h" "$(@D)/python_include/internal/pycore_context.h" && cp -f "c:/Python38/include/internal/pycore_fileutils.h" "$(@D)/python_include/internal/pycore_fileutils.h" && cp -f "c:/Python38/include/internal/pycore_getopt.h" "$(@D)/python_include/internal/pycore_getopt.h" && cp -f "c:/Python38/include/internal/pycore_gil.h" "$(@D)/python_include/internal/pycore_gil.h" && cp -f "c:/Python38/include/internal/pycore_hamt.h" "$(@D)/python_include/internal/pycore_hamt.h" && cp -f "c:/Python38/include/internal/pycore_initconfig.h" "$(@D)/python_include/internal/pycore_initconfig.h" && cp -f "c:/Python38/include/internal/pycore_object.h" "$(@D)/python_include/internal/pycore_object.h" && cp -f "c:/Python38/include/internal/pycore_pathconfig.h" "$(@D)/python_include/internal/pycore_pathconfig.h" && cp -f "c:/Python38/include/internal/pycore_pyerrors.h" "$(@D)/python_include/internal/pycore_pyerrors.h" && cp -f "c:/Python38/include/internal/pycore_pyhash.h" "$(@D)/python_include/internal/pycore_pyhash.h" && cp -f "c:/Python38/include/internal/pycore_pylifecycle.h" "$(@D)/python_include/internal/pycore_pylifecycle.h" && cp -f "c:/Python38/include/internal/pycore_pymem.h" "$(@D)/python_include/internal/pycore_pymem.h" && cp -f "c:/Python38/include/internal/pycore_pystate.h" "$(@D)/python_include/internal/pycore_pystate.h" && cp -f "c:/Python38/include/internal/pycore_traceback.h" "$(@D)/python_include/internal/pycore_traceback.h" && cp -f "c:/Python38/include/internal/pycore_tupleobject.h" "$(@D)/python_include/internal/pycore_tupleobject.h" && cp -f "c:/Python38/include/internal/pycore_warnings.h" "$(@D)/python_include/internal/pycore_warnings.h" && cp -f "c:/Python38/include/interpreteridobject.h" "$(@D)/python_include/interpreteridobject.h" && cp -f "c:/Python38/include/intrcheck.h" "$(@D)/python_include/intrcheck.h" && cp -f "c:/Python38/include/iterobject.h" "$(@D)/python_include/iterobject.h" && cp -f "c:/Python38/include/listobject.h" "$(@D)/python_include/listobject.h" && cp -f "c:/Python38/include/longintrepr.h" "$(@D)/python_include/longintrepr.h" && cp -f "c:/Python38/include/longobject.h" "$(@D)/python_include/longobject.h" && cp -f "c:/Python38/include/marshal.h" "$(@D)/python_include/marshal.h" && cp -f "c:/Python38/include/memoryobject.h" "$(@D)/python_include/memoryobject.h" && cp -f "c:/Python38/include/methodobject.h" "$(@D)/python_include/methodobject.h" && cp -f "c:/Python38/include/modsupport.h" "$(@D)/python_include/modsupport.h" && cp -f "c:/Python38/include/moduleobject.h" "$(@D)/python_include/moduleobject.h" && cp -f "c:/Python38/include/namespaceobject.h" "$(@D)/python_include/namespaceobject.h" && cp -f "c:/Python38/include/node.h" "$(@D)/python_include/node.h" && cp -f "c:/Python38/include/object.h" "$(@D)/python_include/object.h" && cp -f "c:/Python38/include/objimpl.h" "$(@D)/python_include/objimpl.h" && cp -f "c:/Python38/include/odictobject.h" "$(@D)/python_include/odictobject.h" && cp -f "c:/Python38/include/opcode.h" "$(@D)/python_include/opcode.h" && cp -f "c:/Python38/include/osdefs.h" "$(@D)/python_include/osdefs.h" && cp -f "c:/Python38/include/osmodule.h" "$(@D)/python_include/osmodule.h" && cp -f "c:/Python38/include/parsetok.h" "$(@D)/python_include/parsetok.h" && cp -f "c:/Python38/include/patchlevel.h" "$(@D)/python_include/patchlevel.h" && cp -f "c:/Python38/include/picklebufobject.h" "$(@D)/python_include/picklebufobject.h" && cp -f "c:/Python38/include/py_curses.h" "$(@D)/python_include/py_curses.h" && cp -f "c:/Python38/include/pyarena.h" "$(@D)/python_include/pyarena.h" && cp -f "c:/Python38/include/pycapsule.h" "$(@D)/python_include/pycapsule.h" && cp -f "c:/Python38/include/pyconfig.h" "$(@D)/python_include/pyconfig.h" && cp -f "c:/Python38/include/pyctype.h" "$(@D)/python_include/pyctype.h" && cp -f "c:/Python38/include/pydebug.h" "$(@D)/python_include/pydebug.h" && cp -f "c:/Python38/include/pydtrace.h" "$(@D)/python_include/pydtrace.h" && cp -f "c:/Python38/include/pyerrors.h" "$(@D)/python_include/pyerrors.h" && cp -f "c:/Python38/include/pyexpat.h" "$(@D)/python_include/pyexpat.h" && cp -f "c:/Python38/include/pyfpe.h" "$(@D)/python_include/pyfpe.h" && cp -f "c:/Python38/include/pyhash.h" "$(@D)/python_include/pyhash.h" && cp -f "c:/Python38/include/pylifecycle.h" "$(@D)/python_include/pylifecycle.h" && cp -f "c:/Python38/include/pymacconfig.h" "$(@D)/python_include/pymacconfig.h" && cp -f "c:/Python38/include/pymacro.h" "$(@D)/python_include/pymacro.h" && cp -f "c:/Python38/include/pymath.h" "$(@D)/python_include/pymath.h" && cp -f "c:/Python38/include/pymem.h" "$(@D)/python_include/pymem.h" && cp -f "c:/Python38/include/pyport.h" "$(@D)/python_include/pyport.h" && cp -f "c:/Python38/include/pystate.h" "$(@D)/python_include/pystate.h" && cp -f "c:/Python38/include/pystrcmp.h" "$(@D)/python_include/pystrcmp.h" && cp -f "c:/Python38/include/pystrhex.h" "$(@D)/python_include/pystrhex.h" && cp -f "c:/Python38/include/pystrtod.h" "$(@D)/python_include/pystrtod.h" && cp -f "c:/Python38/include/pythonrun.h" "$(@D)/python_include/pythonrun.h" && cp -f "c:/Python38/include/pythread.h" "$(@D)/python_include/pythread.h" && cp -f "c:/Python38/include/pytime.h" "$(@D)/python_include/pytime.h" && cp -f "c:/Python38/include/rangeobject.h" "$(@D)/python_include/rangeobject.h" && cp -f "c:/Python38/include/setobject.h" "$(@D)/python_include/setobject.h" && cp -f "c:/Python38/include/sliceobject.h" "$(@D)/python_include/sliceobject.h" && cp -f "c:/Python38/include/structmember.h" "$(@D)/python_include/structmember.h" && cp -f "c:/Python38/include/structseq.h" "$(@D)/python_include/structseq.h" && cp -f "c:/Python38/include/symtable.h" "$(@D)/python_include/symtable.h" && cp -f "c:/Python38/include/sysmodule.h" "$(@D)/python_include/sysmodule.h" && cp -f "c:/Python38/include/token.h" "$(@D)/python_include/token.h" && cp -f "c:/Python38/include/traceback.h" "$(@D)/python_include/traceback.h" && cp -f "c:/Python38/include/tracemalloc.h" "$(@D)/python_include/tracemalloc.h" && cp -f "c:/Python38/include/tupleobject.h" "$(@D)/python_include/tupleobject.h" && cp -f "c:/Python38/include/typeslots.h" "$(@D)/python_include/typeslots.h" && cp -f "c:/Python38/include/ucnhash.h" "$(@D)/python_include/ucnhash.h" && cp -f "c:/Python38/include/unicodeobject.h" "$(@D)/python_include/unicodeobject.h" && cp -f "c:/Python38/include/warnings.h" "$(@D)/python_include/warnings.h" && cp -f "c:/Python38/include/weakrefobject.h" "$(@D)/python_include/weakrefobject.h"
   """,
)

genrule(
    name = "numpy_include",
    outs = [
        "numpy_include/numpy/__multiarray_api.h",
        "numpy_include/numpy/__ufunc_api.h",
        "numpy_include/numpy/_neighborhood_iterator_imp.h",
        "numpy_include/numpy/_numpyconfig.h",
        "numpy_include/numpy/arrayobject.h",
        "numpy_include/numpy/arrayscalars.h",
        "numpy_include/numpy/halffloat.h",
        "numpy_include/numpy/multiarray_api.txt",
        "numpy_include/numpy/ndarrayobject.h",
        "numpy_include/numpy/ndarraytypes.h",
        "numpy_include/numpy/noprefix.h",
        "numpy_include/numpy/npy_1_7_deprecated_api.h",
        "numpy_include/numpy/npy_3kcompat.h",
        "numpy_include/numpy/npy_common.h",
        "numpy_include/numpy/npy_cpu.h",
        "numpy_include/numpy/npy_endian.h",
        "numpy_include/numpy/npy_interrupt.h",
        "numpy_include/numpy/npy_math.h",
        "numpy_include/numpy/npy_no_deprecated_api.h",
        "numpy_include/numpy/npy_os.h",
        "numpy_include/numpy/numpyconfig.h",
        "numpy_include/numpy/old_defines.h",
        "numpy_include/numpy/oldnumeric.h",
        "numpy_include/numpy/random/bitgen.h",
        "numpy_include/numpy/random/distributions.h",
        "numpy_include/numpy/ufunc_api.txt",
        "numpy_include/numpy/ufuncobject.h",
        "numpy_include/numpy/utils.h",
    ],
    cmd = """
cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/__multiarray_api.h" "$(@D)/numpy_include/numpy/__multiarray_api.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/__ufunc_api.h" "$(@D)/numpy_include/numpy/__ufunc_api.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/_neighborhood_iterator_imp.h" "$(@D)/numpy_include/numpy/_neighborhood_iterator_imp.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/_numpyconfig.h" "$(@D)/numpy_include/numpy/_numpyconfig.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/arrayobject.h" "$(@D)/numpy_include/numpy/arrayobject.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/arrayscalars.h" "$(@D)/numpy_include/numpy/arrayscalars.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/halffloat.h" "$(@D)/numpy_include/numpy/halffloat.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/multiarray_api.txt" "$(@D)/numpy_include/numpy/multiarray_api.txt" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/ndarrayobject.h" "$(@D)/numpy_include/numpy/ndarrayobject.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/ndarraytypes.h" "$(@D)/numpy_include/numpy/ndarraytypes.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/noprefix.h" "$(@D)/numpy_include/numpy/noprefix.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h" "$(@D)/numpy_include/numpy/npy_1_7_deprecated_api.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/npy_3kcompat.h" "$(@D)/numpy_include/numpy/npy_3kcompat.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/npy_common.h" "$(@D)/numpy_include/numpy/npy_common.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/npy_cpu.h" "$(@D)/numpy_include/numpy/npy_cpu.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/npy_endian.h" "$(@D)/numpy_include/numpy/npy_endian.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/npy_interrupt.h" "$(@D)/numpy_include/numpy/npy_interrupt.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/npy_math.h" "$(@D)/numpy_include/numpy/npy_math.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/npy_no_deprecated_api.h" "$(@D)/numpy_include/numpy/npy_no_deprecated_api.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/npy_os.h" "$(@D)/numpy_include/numpy/npy_os.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/numpyconfig.h" "$(@D)/numpy_include/numpy/numpyconfig.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/old_defines.h" "$(@D)/numpy_include/numpy/old_defines.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/oldnumeric.h" "$(@D)/numpy_include/numpy/oldnumeric.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/random/bitgen.h" "$(@D)/numpy_include/numpy/random/bitgen.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/random/distributions.h" "$(@D)/numpy_include/numpy/random/distributions.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/ufunc_api.txt" "$(@D)/numpy_include/numpy/ufunc_api.txt" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/ufuncobject.h" "$(@D)/numpy_include/numpy/ufuncobject.h" && cp -f "c:/Python38/lib/site-packages/numpy/core/include/numpy/utils.h" "$(@D)/numpy_include/numpy/utils.h"
   """,
)

genrule(
    name = "python_import_lib",
    outs = [
        "python38.lib",
    ],
    cmd = """
cp -f "c:/Python38/libs/python38.lib" "$(@D)/python38.lib"
   """,
)
