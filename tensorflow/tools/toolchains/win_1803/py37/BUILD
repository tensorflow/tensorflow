# Point both runtimes to the same python binary to ensure we always
# use the python binary specified by ./configure.py script.
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_runtime(
    name = "py3_runtime",
    interpreter_path = "C:/Python37/python.exe",
    python_version = "PY3",
)

py_runtime_pair(
    name = "py_runtime_pair",
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
        "python_include/accu.h",
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
        "python_include/internal/ceval.h",
        "python_include/internal/condvar.h",
        "python_include/internal/context.h",
        "python_include/internal/gil.h",
        "python_include/internal/hamt.h",
        "python_include/internal/hash.h",
        "python_include/internal/import.h",
        "python_include/internal/mem.h",
        "python_include/internal/pygetopt.h",
        "python_include/internal/pystate.h",
        "python_include/internal/warnings.h",
        "python_include/intrcheck.h",
        "python_include/iterobject.h",
        "python_include/listobject.h",
        "python_include/longintrepr.h",
        "python_include/longobject.h",
        "python_include/marshal.h",
        "python_include/memoryobject.h",
        "python_include/metagrammar.h",
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
        "python_include/pgen.h",
        "python_include/pgenheaders.h",
        "python_include/py_curses.h",
        "python_include/pyarena.h",
        "python_include/pyatomic.h",
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
        "python_include/tupleobject.h",
        "python_include/typeslots.h",
        "python_include/ucnhash.h",
        "python_include/unicodeobject.h",
        "python_include/warnings.h",
        "python_include/weakrefobject.h",
    ],
    cmd = """
cp -f "c:/python37/include/Python-ast.h" "$(@D)/python_include/Python-ast.h" && cp -f "c:/python37/include/Python.h" "$(@D)/python_include/Python.h" && cp -f "c:/python37/include/abstract.h" "$(@D)/python_include/abstract.h" && cp -f "c:/python37/include/accu.h" "$(@D)/python_include/accu.h" && cp -f "c:/python37/include/asdl.h" "$(@D)/python_include/asdl.h" && cp -f "c:/python37/include/ast.h" "$(@D)/python_include/ast.h" && cp -f "c:/python37/include/bitset.h" "$(@D)/python_include/bitset.h" && cp -f "c:/python37/include/bltinmodule.h" "$(@D)/python_include/bltinmodule.h" && cp -f "c:/python37/include/boolobject.h" "$(@D)/python_include/boolobject.h" && cp -f "c:/python37/include/bytearrayobject.h" "$(@D)/python_include/bytearrayobject.h" && cp -f "c:/python37/include/bytes_methods.h" "$(@D)/python_include/bytes_methods.h" && cp -f "c:/python37/include/bytesobject.h" "$(@D)/python_include/bytesobject.h" && cp -f "c:/python37/include/cellobject.h" "$(@D)/python_include/cellobject.h" && cp -f "c:/python37/include/ceval.h" "$(@D)/python_include/ceval.h" && cp -f "c:/python37/include/classobject.h" "$(@D)/python_include/classobject.h" && cp -f "c:/python37/include/code.h" "$(@D)/python_include/code.h" && cp -f "c:/python37/include/codecs.h" "$(@D)/python_include/codecs.h" && cp -f "c:/python37/include/compile.h" "$(@D)/python_include/compile.h" && cp -f "c:/python37/include/complexobject.h" "$(@D)/python_include/complexobject.h" && cp -f "c:/python37/include/context.h" "$(@D)/python_include/context.h" && cp -f "c:/python37/include/datetime.h" "$(@D)/python_include/datetime.h" && cp -f "c:/python37/include/descrobject.h" "$(@D)/python_include/descrobject.h" && cp -f "c:/python37/include/dictobject.h" "$(@D)/python_include/dictobject.h" && cp -f "c:/python37/include/dtoa.h" "$(@D)/python_include/dtoa.h" && cp -f "c:/python37/include/dynamic_annotations.h" "$(@D)/python_include/dynamic_annotations.h" && cp -f "c:/python37/include/enumobject.h" "$(@D)/python_include/enumobject.h" && cp -f "c:/python37/include/errcode.h" "$(@D)/python_include/errcode.h" && cp -f "c:/python37/include/eval.h" "$(@D)/python_include/eval.h" && cp -f "c:/python37/include/fileobject.h" "$(@D)/python_include/fileobject.h" && cp -f "c:/python37/include/fileutils.h" "$(@D)/python_include/fileutils.h" && cp -f "c:/python37/include/floatobject.h" "$(@D)/python_include/floatobject.h" && cp -f "c:/python37/include/frameobject.h" "$(@D)/python_include/frameobject.h" && cp -f "c:/python37/include/funcobject.h" "$(@D)/python_include/funcobject.h" && cp -f "c:/python37/include/genobject.h" "$(@D)/python_include/genobject.h" && cp -f "c:/python37/include/graminit.h" "$(@D)/python_include/graminit.h" && cp -f "c:/python37/include/grammar.h" "$(@D)/python_include/grammar.h" && cp -f "c:/python37/include/import.h" "$(@D)/python_include/import.h" && cp -f "c:/python37/include/internal/ceval.h" "$(@D)/python_include/internal/ceval.h" && cp -f "c:/python37/include/internal/condvar.h" "$(@D)/python_include/internal/condvar.h" && cp -f "c:/python37/include/internal/context.h" "$(@D)/python_include/internal/context.h" && cp -f "c:/python37/include/internal/gil.h" "$(@D)/python_include/internal/gil.h" && cp -f "c:/python37/include/internal/hamt.h" "$(@D)/python_include/internal/hamt.h" && cp -f "c:/python37/include/internal/hash.h" "$(@D)/python_include/internal/hash.h" && cp -f "c:/python37/include/internal/import.h" "$(@D)/python_include/internal/import.h" && cp -f "c:/python37/include/internal/mem.h" "$(@D)/python_include/internal/mem.h" && cp -f "c:/python37/include/internal/pygetopt.h" "$(@D)/python_include/internal/pygetopt.h" && cp -f "c:/python37/include/internal/pystate.h" "$(@D)/python_include/internal/pystate.h" && cp -f "c:/python37/include/internal/warnings.h" "$(@D)/python_include/internal/warnings.h" && cp -f "c:/python37/include/intrcheck.h" "$(@D)/python_include/intrcheck.h" && cp -f "c:/python37/include/iterobject.h" "$(@D)/python_include/iterobject.h" && cp -f "c:/python37/include/listobject.h" "$(@D)/python_include/listobject.h" && cp -f "c:/python37/include/longintrepr.h" "$(@D)/python_include/longintrepr.h" && cp -f "c:/python37/include/longobject.h" "$(@D)/python_include/longobject.h" && cp -f "c:/python37/include/marshal.h" "$(@D)/python_include/marshal.h" && cp -f "c:/python37/include/memoryobject.h" "$(@D)/python_include/memoryobject.h" && cp -f "c:/python37/include/metagrammar.h" "$(@D)/python_include/metagrammar.h" && cp -f "c:/python37/include/methodobject.h" "$(@D)/python_include/methodobject.h" && cp -f "c:/python37/include/modsupport.h" "$(@D)/python_include/modsupport.h" && cp -f "c:/python37/include/moduleobject.h" "$(@D)/python_include/moduleobject.h" && cp -f "c:/python37/include/namespaceobject.h" "$(@D)/python_include/namespaceobject.h" && cp -f "c:/python37/include/node.h" "$(@D)/python_include/node.h" && cp -f "c:/python37/include/object.h" "$(@D)/python_include/object.h" && cp -f "c:/python37/include/objimpl.h" "$(@D)/python_include/objimpl.h" && cp -f "c:/python37/include/odictobject.h" "$(@D)/python_include/odictobject.h" && cp -f "c:/python37/include/opcode.h" "$(@D)/python_include/opcode.h" && cp -f "c:/python37/include/osdefs.h" "$(@D)/python_include/osdefs.h" && cp -f "c:/python37/include/osmodule.h" "$(@D)/python_include/osmodule.h" && cp -f "c:/python37/include/parsetok.h" "$(@D)/python_include/parsetok.h" && cp -f "c:/python37/include/patchlevel.h" "$(@D)/python_include/patchlevel.h" && cp -f "c:/python37/include/pgen.h" "$(@D)/python_include/pgen.h" && cp -f "c:/python37/include/pgenheaders.h" "$(@D)/python_include/pgenheaders.h" && cp -f "c:/python37/include/py_curses.h" "$(@D)/python_include/py_curses.h" && cp -f "c:/python37/include/pyarena.h" "$(@D)/python_include/pyarena.h" && cp -f "c:/python37/include/pyatomic.h" "$(@D)/python_include/pyatomic.h" && cp -f "c:/python37/include/pycapsule.h" "$(@D)/python_include/pycapsule.h" && cp -f "c:/python37/include/pyconfig.h" "$(@D)/python_include/pyconfig.h" && cp -f "c:/python37/include/pyctype.h" "$(@D)/python_include/pyctype.h" && cp -f "c:/python37/include/pydebug.h" "$(@D)/python_include/pydebug.h" && cp -f "c:/python37/include/pydtrace.h" "$(@D)/python_include/pydtrace.h" && cp -f "c:/python37/include/pyerrors.h" "$(@D)/python_include/pyerrors.h" && cp -f "c:/python37/include/pyexpat.h" "$(@D)/python_include/pyexpat.h" && cp -f "c:/python37/include/pyfpe.h" "$(@D)/python_include/pyfpe.h" && cp -f "c:/python37/include/pyhash.h" "$(@D)/python_include/pyhash.h" && cp -f "c:/python37/include/pylifecycle.h" "$(@D)/python_include/pylifecycle.h" && cp -f "c:/python37/include/pymacconfig.h" "$(@D)/python_include/pymacconfig.h" && cp -f "c:/python37/include/pymacro.h" "$(@D)/python_include/pymacro.h" && cp -f "c:/python37/include/pymath.h" "$(@D)/python_include/pymath.h" && cp -f "c:/python37/include/pymem.h" "$(@D)/python_include/pymem.h" && cp -f "c:/python37/include/pyport.h" "$(@D)/python_include/pyport.h" && cp -f "c:/python37/include/pystate.h" "$(@D)/python_include/pystate.h" && cp -f "c:/python37/include/pystrcmp.h" "$(@D)/python_include/pystrcmp.h" && cp -f "c:/python37/include/pystrhex.h" "$(@D)/python_include/pystrhex.h" && cp -f "c:/python37/include/pystrtod.h" "$(@D)/python_include/pystrtod.h" && cp -f "c:/python37/include/pythonrun.h" "$(@D)/python_include/pythonrun.h" && cp -f "c:/python37/include/pythread.h" "$(@D)/python_include/pythread.h" && cp -f "c:/python37/include/pytime.h" "$(@D)/python_include/pytime.h" && cp -f "c:/python37/include/rangeobject.h" "$(@D)/python_include/rangeobject.h" && cp -f "c:/python37/include/setobject.h" "$(@D)/python_include/setobject.h" && cp -f "c:/python37/include/sliceobject.h" "$(@D)/python_include/sliceobject.h" && cp -f "c:/python37/include/structmember.h" "$(@D)/python_include/structmember.h" && cp -f "c:/python37/include/structseq.h" "$(@D)/python_include/structseq.h" && cp -f "c:/python37/include/symtable.h" "$(@D)/python_include/symtable.h" && cp -f "c:/python37/include/sysmodule.h" "$(@D)/python_include/sysmodule.h" && cp -f "c:/python37/include/token.h" "$(@D)/python_include/token.h" && cp -f "c:/python37/include/traceback.h" "$(@D)/python_include/traceback.h" && cp -f "c:/python37/include/tupleobject.h" "$(@D)/python_include/tupleobject.h" && cp -f "c:/python37/include/typeslots.h" "$(@D)/python_include/typeslots.h" && cp -f "c:/python37/include/ucnhash.h" "$(@D)/python_include/ucnhash.h" && cp -f "c:/python37/include/unicodeobject.h" "$(@D)/python_include/unicodeobject.h" && cp -f "c:/python37/include/warnings.h" "$(@D)/python_include/warnings.h" && cp -f "c:/python37/include/weakrefobject.h" "$(@D)/python_include/weakrefobject.h"
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
        "numpy_include/numpy/ufunc_api.txt",
        "numpy_include/numpy/ufuncobject.h",
        "numpy_include/numpy/utils.h",
    ],
    cmd = """
cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/__multiarray_api.h" "$(@D)/numpy_include/numpy/__multiarray_api.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/__ufunc_api.h" "$(@D)/numpy_include/numpy/__ufunc_api.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/_neighborhood_iterator_imp.h" "$(@D)/numpy_include/numpy/_neighborhood_iterator_imp.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/_numpyconfig.h" "$(@D)/numpy_include/numpy/_numpyconfig.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/arrayobject.h" "$(@D)/numpy_include/numpy/arrayobject.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/arrayscalars.h" "$(@D)/numpy_include/numpy/arrayscalars.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/halffloat.h" "$(@D)/numpy_include/numpy/halffloat.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/multiarray_api.txt" "$(@D)/numpy_include/numpy/multiarray_api.txt" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/ndarrayobject.h" "$(@D)/numpy_include/numpy/ndarrayobject.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/ndarraytypes.h" "$(@D)/numpy_include/numpy/ndarraytypes.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/noprefix.h" "$(@D)/numpy_include/numpy/noprefix.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h" "$(@D)/numpy_include/numpy/npy_1_7_deprecated_api.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/npy_3kcompat.h" "$(@D)/numpy_include/numpy/npy_3kcompat.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/npy_common.h" "$(@D)/numpy_include/numpy/npy_common.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/npy_cpu.h" "$(@D)/numpy_include/numpy/npy_cpu.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/npy_endian.h" "$(@D)/numpy_include/numpy/npy_endian.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/npy_interrupt.h" "$(@D)/numpy_include/numpy/npy_interrupt.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/npy_math.h" "$(@D)/numpy_include/numpy/npy_math.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/npy_no_deprecated_api.h" "$(@D)/numpy_include/numpy/npy_no_deprecated_api.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/npy_os.h" "$(@D)/numpy_include/numpy/npy_os.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/numpyconfig.h" "$(@D)/numpy_include/numpy/numpyconfig.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/old_defines.h" "$(@D)/numpy_include/numpy/old_defines.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/oldnumeric.h" "$(@D)/numpy_include/numpy/oldnumeric.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/ufunc_api.txt" "$(@D)/numpy_include/numpy/ufunc_api.txt" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/ufuncobject.h" "$(@D)/numpy_include/numpy/ufuncobject.h" && cp -f "c:/python37/lib/site-packages/numpy/core/include/numpy/utils.h" "$(@D)/numpy_include/numpy/utils.h"
   """,
)

genrule(
    name = "python_import_lib",
    outs = [
        "python37.lib",
    ],
    cmd = """
cp -f "c:/python37/libs/python37.lib" "$(@D)/python37.lib"
   """,
)
