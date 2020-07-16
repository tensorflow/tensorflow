licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# Point both runtimes to the same python binary to ensure we always
# use the python binary specified by ./configure.py script.
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

py_runtime(
    name = "py2_runtime",
    interpreter_path = "/usr/bin/python2",
    python_version = "PY2",
)

py_runtime(
    name = "py3_runtime",
    interpreter_path = "/usr/bin/python2",
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
        "python_include/boolobject.h",
        "python_include/bufferobject.h",
        "python_include/bytearrayobject.h",
        "python_include/bytes_methods.h",
        "python_include/bytesobject.h",
        "python_include/cStringIO.h",
        "python_include/cellobject.h",
        "python_include/ceval.h",
        "python_include/classobject.h",
        "python_include/cobject.h",
        "python_include/code.h",
        "python_include/codecs.h",
        "python_include/compile.h",
        "python_include/complexobject.h",
        "python_include/datetime.h",
        "python_include/descrobject.h",
        "python_include/dictobject.h",
        "python_include/dtoa.h",
        "python_include/enumobject.h",
        "python_include/errcode.h",
        "python_include/eval.h",
        "python_include/fileobject.h",
        "python_include/floatobject.h",
        "python_include/frameobject.h",
        "python_include/funcobject.h",
        "python_include/genobject.h",
        "python_include/graminit.h",
        "python_include/grammar.h",
        "python_include/import.h",
        "python_include/intobject.h",
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
        "python_include/node.h",
        "python_include/object.h",
        "python_include/objimpl.h",
        "python_include/opcode.h",
        "python_include/osdefs.h",
        "python_include/parsetok.h",
        "python_include/patchlevel.h",
        "python_include/pgen.h",
        "python_include/pgenheaders.h",
        "python_include/py_curses.h",
        "python_include/pyarena.h",
        "python_include/pycapsule.h",
        "python_include/pyconfig.h",
        "python_include/pyctype.h",
        "python_include/pydebug.h",
        "python_include/pyerrors.h",
        "python_include/pyexpat.h",
        "python_include/pyfpe.h",
        "python_include/pygetopt.h",
        "python_include/pymacconfig.h",
        "python_include/pymactoolbox.h",
        "python_include/pymath.h",
        "python_include/pymem.h",
        "python_include/pyport.h",
        "python_include/pystate.h",
        "python_include/pystrcmp.h",
        "python_include/pystrtod.h",
        "python_include/pythonrun.h",
        "python_include/pythread.h",
        "python_include/rangeobject.h",
        "python_include/setobject.h",
        "python_include/sliceobject.h",
        "python_include/stringobject.h",
        "python_include/structmember.h",
        "python_include/structseq.h",
        "python_include/symtable.h",
        "python_include/sysmodule.h",
        "python_include/timefuncs.h",
        "python_include/token.h",
        "python_include/traceback.h",
        "python_include/tupleobject.h",
        "python_include/ucnhash.h",
        "python_include/unicodeobject.h",
        "python_include/warnings.h",
        "python_include/weakrefobject.h",
    ],
    cmd = """
cp -f "/usr/include/python2.7/Python-ast.h" "$(@D)/python_include/Python-ast.h" && cp -f "/usr/include/python2.7/Python.h" "$(@D)/python_include/Python.h" && cp -f "/usr/include/python2.7/abstract.h" "$(@D)/python_include/abstract.h" && cp -f "/usr/include/python2.7/asdl.h" "$(@D)/python_include/asdl.h" && cp -f "/usr/include/python2.7/ast.h" "$(@D)/python_include/ast.h" && cp -f "/usr/include/python2.7/bitset.h" "$(@D)/python_include/bitset.h" && cp -f "/usr/include/python2.7/boolobject.h" "$(@D)/python_include/boolobject.h" && cp -f "/usr/include/python2.7/bufferobject.h" "$(@D)/python_include/bufferobject.h" && cp -f "/usr/include/python2.7/bytearrayobject.h" "$(@D)/python_include/bytearrayobject.h" && cp -f "/usr/include/python2.7/bytes_methods.h" "$(@D)/python_include/bytes_methods.h" && cp -f "/usr/include/python2.7/bytesobject.h" "$(@D)/python_include/bytesobject.h" && cp -f "/usr/include/python2.7/cStringIO.h" "$(@D)/python_include/cStringIO.h" && cp -f "/usr/include/python2.7/cellobject.h" "$(@D)/python_include/cellobject.h" && cp -f "/usr/include/python2.7/ceval.h" "$(@D)/python_include/ceval.h" && cp -f "/usr/include/python2.7/classobject.h" "$(@D)/python_include/classobject.h" && cp -f "/usr/include/python2.7/cobject.h" "$(@D)/python_include/cobject.h" && cp -f "/usr/include/python2.7/code.h" "$(@D)/python_include/code.h" && cp -f "/usr/include/python2.7/codecs.h" "$(@D)/python_include/codecs.h" && cp -f "/usr/include/python2.7/compile.h" "$(@D)/python_include/compile.h" && cp -f "/usr/include/python2.7/complexobject.h" "$(@D)/python_include/complexobject.h" && cp -f "/usr/include/python2.7/datetime.h" "$(@D)/python_include/datetime.h" && cp -f "/usr/include/python2.7/descrobject.h" "$(@D)/python_include/descrobject.h" && cp -f "/usr/include/python2.7/dictobject.h" "$(@D)/python_include/dictobject.h" && cp -f "/usr/include/python2.7/dtoa.h" "$(@D)/python_include/dtoa.h" && cp -f "/usr/include/python2.7/enumobject.h" "$(@D)/python_include/enumobject.h" && cp -f "/usr/include/python2.7/errcode.h" "$(@D)/python_include/errcode.h" && cp -f "/usr/include/python2.7/eval.h" "$(@D)/python_include/eval.h" && cp -f "/usr/include/python2.7/fileobject.h" "$(@D)/python_include/fileobject.h" && cp -f "/usr/include/python2.7/floatobject.h" "$(@D)/python_include/floatobject.h" && cp -f "/usr/include/python2.7/frameobject.h" "$(@D)/python_include/frameobject.h" && cp -f "/usr/include/python2.7/funcobject.h" "$(@D)/python_include/funcobject.h" && cp -f "/usr/include/python2.7/genobject.h" "$(@D)/python_include/genobject.h" && cp -f "/usr/include/python2.7/graminit.h" "$(@D)/python_include/graminit.h" && cp -f "/usr/include/python2.7/grammar.h" "$(@D)/python_include/grammar.h" && cp -f "/usr/include/python2.7/import.h" "$(@D)/python_include/import.h" && cp -f "/usr/include/python2.7/intobject.h" "$(@D)/python_include/intobject.h" && cp -f "/usr/include/python2.7/intrcheck.h" "$(@D)/python_include/intrcheck.h" && cp -f "/usr/include/python2.7/iterobject.h" "$(@D)/python_include/iterobject.h" && cp -f "/usr/include/python2.7/listobject.h" "$(@D)/python_include/listobject.h" && cp -f "/usr/include/python2.7/longintrepr.h" "$(@D)/python_include/longintrepr.h" && cp -f "/usr/include/python2.7/longobject.h" "$(@D)/python_include/longobject.h" && cp -f "/usr/include/python2.7/marshal.h" "$(@D)/python_include/marshal.h" && cp -f "/usr/include/python2.7/memoryobject.h" "$(@D)/python_include/memoryobject.h" && cp -f "/usr/include/python2.7/metagrammar.h" "$(@D)/python_include/metagrammar.h" && cp -f "/usr/include/python2.7/methodobject.h" "$(@D)/python_include/methodobject.h" && cp -f "/usr/include/python2.7/modsupport.h" "$(@D)/python_include/modsupport.h" && cp -f "/usr/include/python2.7/moduleobject.h" "$(@D)/python_include/moduleobject.h" && cp -f "/usr/include/python2.7/node.h" "$(@D)/python_include/node.h" && cp -f "/usr/include/python2.7/object.h" "$(@D)/python_include/object.h" && cp -f "/usr/include/python2.7/objimpl.h" "$(@D)/python_include/objimpl.h" && cp -f "/usr/include/python2.7/opcode.h" "$(@D)/python_include/opcode.h" && cp -f "/usr/include/python2.7/osdefs.h" "$(@D)/python_include/osdefs.h" && cp -f "/usr/include/python2.7/parsetok.h" "$(@D)/python_include/parsetok.h" && cp -f "/usr/include/python2.7/patchlevel.h" "$(@D)/python_include/patchlevel.h" && cp -f "/usr/include/python2.7/pgen.h" "$(@D)/python_include/pgen.h" && cp -f "/usr/include/python2.7/pgenheaders.h" "$(@D)/python_include/pgenheaders.h" && cp -f "/usr/include/python2.7/py_curses.h" "$(@D)/python_include/py_curses.h" && cp -f "/usr/include/python2.7/pyarena.h" "$(@D)/python_include/pyarena.h" && cp -f "/usr/include/python2.7/pycapsule.h" "$(@D)/python_include/pycapsule.h" && cp -f "/usr/include/python2.7/pyconfig.h" "$(@D)/python_include/pyconfig.h" && cp -f "/usr/include/python2.7/pyctype.h" "$(@D)/python_include/pyctype.h" && cp -f "/usr/include/python2.7/pydebug.h" "$(@D)/python_include/pydebug.h" && cp -f "/usr/include/python2.7/pyerrors.h" "$(@D)/python_include/pyerrors.h" && cp -f "/usr/include/python2.7/pyexpat.h" "$(@D)/python_include/pyexpat.h" && cp -f "/usr/include/python2.7/pyfpe.h" "$(@D)/python_include/pyfpe.h" && cp -f "/usr/include/python2.7/pygetopt.h" "$(@D)/python_include/pygetopt.h" && cp -f "/usr/include/python2.7/pymacconfig.h" "$(@D)/python_include/pymacconfig.h" && cp -f "/usr/include/python2.7/pymactoolbox.h" "$(@D)/python_include/pymactoolbox.h" && cp -f "/usr/include/python2.7/pymath.h" "$(@D)/python_include/pymath.h" && cp -f "/usr/include/python2.7/pymem.h" "$(@D)/python_include/pymem.h" && cp -f "/usr/include/python2.7/pyport.h" "$(@D)/python_include/pyport.h" && cp -f "/usr/include/python2.7/pystate.h" "$(@D)/python_include/pystate.h" && cp -f "/usr/include/python2.7/pystrcmp.h" "$(@D)/python_include/pystrcmp.h" && cp -f "/usr/include/python2.7/pystrtod.h" "$(@D)/python_include/pystrtod.h" && cp -f "/usr/include/python2.7/pythonrun.h" "$(@D)/python_include/pythonrun.h" && cp -f "/usr/include/python2.7/pythread.h" "$(@D)/python_include/pythread.h" && cp -f "/usr/include/python2.7/rangeobject.h" "$(@D)/python_include/rangeobject.h" && cp -f "/usr/include/python2.7/setobject.h" "$(@D)/python_include/setobject.h" && cp -f "/usr/include/python2.7/sliceobject.h" "$(@D)/python_include/sliceobject.h" && cp -f "/usr/include/python2.7/stringobject.h" "$(@D)/python_include/stringobject.h" && cp -f "/usr/include/python2.7/structmember.h" "$(@D)/python_include/structmember.h" && cp -f "/usr/include/python2.7/structseq.h" "$(@D)/python_include/structseq.h" && cp -f "/usr/include/python2.7/symtable.h" "$(@D)/python_include/symtable.h" && cp -f "/usr/include/python2.7/sysmodule.h" "$(@D)/python_include/sysmodule.h" && cp -f "/usr/include/python2.7/timefuncs.h" "$(@D)/python_include/timefuncs.h" && cp -f "/usr/include/python2.7/token.h" "$(@D)/python_include/token.h" && cp -f "/usr/include/python2.7/traceback.h" "$(@D)/python_include/traceback.h" && cp -f "/usr/include/python2.7/tupleobject.h" "$(@D)/python_include/tupleobject.h" && cp -f "/usr/include/python2.7/ucnhash.h" "$(@D)/python_include/ucnhash.h" && cp -f "/usr/include/python2.7/unicodeobject.h" "$(@D)/python_include/unicodeobject.h" && cp -f "/usr/include/python2.7/warnings.h" "$(@D)/python_include/warnings.h" && cp -f "/usr/include/python2.7/weakrefobject.h" "$(@D)/python_include/weakrefobject.h"
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
cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h" "$(@D)/numpy_include/numpy/__multiarray_api.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h" "$(@D)/numpy_include/numpy/__ufunc_api.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/_neighborhood_iterator_imp.h" "$(@D)/numpy_include/numpy/_neighborhood_iterator_imp.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/_numpyconfig.h" "$(@D)/numpy_include/numpy/_numpyconfig.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h" "$(@D)/numpy_include/numpy/arrayobject.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayscalars.h" "$(@D)/numpy_include/numpy/arrayscalars.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/halffloat.h" "$(@D)/numpy_include/numpy/halffloat.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/multiarray_api.txt" "$(@D)/numpy_include/numpy/multiarray_api.txt" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h" "$(@D)/numpy_include/numpy/ndarrayobject.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h" "$(@D)/numpy_include/numpy/ndarraytypes.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h" "$(@D)/numpy_include/numpy/noprefix.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h" "$(@D)/numpy_include/numpy/npy_1_7_deprecated_api.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h" "$(@D)/numpy_include/numpy/npy_3kcompat.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_common.h" "$(@D)/numpy_include/numpy/npy_common.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_cpu.h" "$(@D)/numpy_include/numpy/npy_cpu.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_endian.h" "$(@D)/numpy_include/numpy/npy_endian.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_interrupt.h" "$(@D)/numpy_include/numpy/npy_interrupt.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_math.h" "$(@D)/numpy_include/numpy/npy_math.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_no_deprecated_api.h" "$(@D)/numpy_include/numpy/npy_no_deprecated_api.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_os.h" "$(@D)/numpy_include/numpy/npy_os.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/numpyconfig.h" "$(@D)/numpy_include/numpy/numpyconfig.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/old_defines.h" "$(@D)/numpy_include/numpy/old_defines.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/oldnumeric.h" "$(@D)/numpy_include/numpy/oldnumeric.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufunc_api.txt" "$(@D)/numpy_include/numpy/ufunc_api.txt" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h" "$(@D)/numpy_include/numpy/ufuncobject.h" && cp -f "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/utils.h" "$(@D)/numpy_include/numpy/utils.h"
   """,
)
