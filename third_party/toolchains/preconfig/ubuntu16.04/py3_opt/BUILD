licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# Point both runtimes to the same python binary to ensure we always
# use the python binary specified by ./configure.py script.
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

py_runtime(
    name = "py2_runtime",
    interpreter_path = "/usr/bin/python3",
    python_version = "PY2",
)

py_runtime(
    name = "py3_runtime",
    interpreter_path = "/usr/bin/python3",
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
        "python_include/pygetopt.h",
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
cp -f "/opt/python3.6/include/python3.6m/Python-ast.h" "$(@D)/python_include/Python-ast.h" && cp -f "/opt/python3.6/include/python3.6m/Python.h" "$(@D)/python_include/Python.h" && cp -f "/opt/python3.6/include/python3.6m/abstract.h" "$(@D)/python_include/abstract.h" && cp -f "/opt/python3.6/include/python3.6m/accu.h" "$(@D)/python_include/accu.h" && cp -f "/opt/python3.6/include/python3.6m/asdl.h" "$(@D)/python_include/asdl.h" && cp -f "/opt/python3.6/include/python3.6m/ast.h" "$(@D)/python_include/ast.h" && cp -f "/opt/python3.6/include/python3.6m/bitset.h" "$(@D)/python_include/bitset.h" && cp -f "/opt/python3.6/include/python3.6m/bltinmodule.h" "$(@D)/python_include/bltinmodule.h" && cp -f "/opt/python3.6/include/python3.6m/boolobject.h" "$(@D)/python_include/boolobject.h" && cp -f "/opt/python3.6/include/python3.6m/bytearrayobject.h" "$(@D)/python_include/bytearrayobject.h" && cp -f "/opt/python3.6/include/python3.6m/bytes_methods.h" "$(@D)/python_include/bytes_methods.h" && cp -f "/opt/python3.6/include/python3.6m/bytesobject.h" "$(@D)/python_include/bytesobject.h" && cp -f "/opt/python3.6/include/python3.6m/cellobject.h" "$(@D)/python_include/cellobject.h" && cp -f "/opt/python3.6/include/python3.6m/ceval.h" "$(@D)/python_include/ceval.h" && cp -f "/opt/python3.6/include/python3.6m/classobject.h" "$(@D)/python_include/classobject.h" && cp -f "/opt/python3.6/include/python3.6m/code.h" "$(@D)/python_include/code.h" && cp -f "/opt/python3.6/include/python3.6m/codecs.h" "$(@D)/python_include/codecs.h" && cp -f "/opt/python3.6/include/python3.6m/compile.h" "$(@D)/python_include/compile.h" && cp -f "/opt/python3.6/include/python3.6m/complexobject.h" "$(@D)/python_include/complexobject.h" && cp -f "/opt/python3.6/include/python3.6m/datetime.h" "$(@D)/python_include/datetime.h" && cp -f "/opt/python3.6/include/python3.6m/descrobject.h" "$(@D)/python_include/descrobject.h" && cp -f "/opt/python3.6/include/python3.6m/dictobject.h" "$(@D)/python_include/dictobject.h" && cp -f "/opt/python3.6/include/python3.6m/dtoa.h" "$(@D)/python_include/dtoa.h" && cp -f "/opt/python3.6/include/python3.6m/dynamic_annotations.h" "$(@D)/python_include/dynamic_annotations.h" && cp -f "/opt/python3.6/include/python3.6m/enumobject.h" "$(@D)/python_include/enumobject.h" && cp -f "/opt/python3.6/include/python3.6m/errcode.h" "$(@D)/python_include/errcode.h" && cp -f "/opt/python3.6/include/python3.6m/eval.h" "$(@D)/python_include/eval.h" && cp -f "/opt/python3.6/include/python3.6m/fileobject.h" "$(@D)/python_include/fileobject.h" && cp -f "/opt/python3.6/include/python3.6m/fileutils.h" "$(@D)/python_include/fileutils.h" && cp -f "/opt/python3.6/include/python3.6m/floatobject.h" "$(@D)/python_include/floatobject.h" && cp -f "/opt/python3.6/include/python3.6m/frameobject.h" "$(@D)/python_include/frameobject.h" && cp -f "/opt/python3.6/include/python3.6m/funcobject.h" "$(@D)/python_include/funcobject.h" && cp -f "/opt/python3.6/include/python3.6m/genobject.h" "$(@D)/python_include/genobject.h" && cp -f "/opt/python3.6/include/python3.6m/graminit.h" "$(@D)/python_include/graminit.h" && cp -f "/opt/python3.6/include/python3.6m/grammar.h" "$(@D)/python_include/grammar.h" && cp -f "/opt/python3.6/include/python3.6m/import.h" "$(@D)/python_include/import.h" && cp -f "/opt/python3.6/include/python3.6m/intrcheck.h" "$(@D)/python_include/intrcheck.h" && cp -f "/opt/python3.6/include/python3.6m/iterobject.h" "$(@D)/python_include/iterobject.h" && cp -f "/opt/python3.6/include/python3.6m/listobject.h" "$(@D)/python_include/listobject.h" && cp -f "/opt/python3.6/include/python3.6m/longintrepr.h" "$(@D)/python_include/longintrepr.h" && cp -f "/opt/python3.6/include/python3.6m/longobject.h" "$(@D)/python_include/longobject.h" && cp -f "/opt/python3.6/include/python3.6m/marshal.h" "$(@D)/python_include/marshal.h" && cp -f "/opt/python3.6/include/python3.6m/memoryobject.h" "$(@D)/python_include/memoryobject.h" && cp -f "/opt/python3.6/include/python3.6m/metagrammar.h" "$(@D)/python_include/metagrammar.h" && cp -f "/opt/python3.6/include/python3.6m/methodobject.h" "$(@D)/python_include/methodobject.h" && cp -f "/opt/python3.6/include/python3.6m/modsupport.h" "$(@D)/python_include/modsupport.h" && cp -f "/opt/python3.6/include/python3.6m/moduleobject.h" "$(@D)/python_include/moduleobject.h" && cp -f "/opt/python3.6/include/python3.6m/namespaceobject.h" "$(@D)/python_include/namespaceobject.h" && cp -f "/opt/python3.6/include/python3.6m/node.h" "$(@D)/python_include/node.h" && cp -f "/opt/python3.6/include/python3.6m/object.h" "$(@D)/python_include/object.h" && cp -f "/opt/python3.6/include/python3.6m/objimpl.h" "$(@D)/python_include/objimpl.h" && cp -f "/opt/python3.6/include/python3.6m/odictobject.h" "$(@D)/python_include/odictobject.h" && cp -f "/opt/python3.6/include/python3.6m/opcode.h" "$(@D)/python_include/opcode.h" && cp -f "/opt/python3.6/include/python3.6m/osdefs.h" "$(@D)/python_include/osdefs.h" && cp -f "/opt/python3.6/include/python3.6m/osmodule.h" "$(@D)/python_include/osmodule.h" && cp -f "/opt/python3.6/include/python3.6m/parsetok.h" "$(@D)/python_include/parsetok.h" && cp -f "/opt/python3.6/include/python3.6m/patchlevel.h" "$(@D)/python_include/patchlevel.h" && cp -f "/opt/python3.6/include/python3.6m/pgen.h" "$(@D)/python_include/pgen.h" && cp -f "/opt/python3.6/include/python3.6m/pgenheaders.h" "$(@D)/python_include/pgenheaders.h" && cp -f "/opt/python3.6/include/python3.6m/py_curses.h" "$(@D)/python_include/py_curses.h" && cp -f "/opt/python3.6/include/python3.6m/pyarena.h" "$(@D)/python_include/pyarena.h" && cp -f "/opt/python3.6/include/python3.6m/pyatomic.h" "$(@D)/python_include/pyatomic.h" && cp -f "/opt/python3.6/include/python3.6m/pycapsule.h" "$(@D)/python_include/pycapsule.h" && cp -f "/opt/python3.6/include/python3.6m/pyconfig.h" "$(@D)/python_include/pyconfig.h" && cp -f "/opt/python3.6/include/python3.6m/pyctype.h" "$(@D)/python_include/pyctype.h" && cp -f "/opt/python3.6/include/python3.6m/pydebug.h" "$(@D)/python_include/pydebug.h" && cp -f "/opt/python3.6/include/python3.6m/pydtrace.h" "$(@D)/python_include/pydtrace.h" && cp -f "/opt/python3.6/include/python3.6m/pyerrors.h" "$(@D)/python_include/pyerrors.h" && cp -f "/opt/python3.6/include/python3.6m/pyexpat.h" "$(@D)/python_include/pyexpat.h" && cp -f "/opt/python3.6/include/python3.6m/pyfpe.h" "$(@D)/python_include/pyfpe.h" && cp -f "/opt/python3.6/include/python3.6m/pygetopt.h" "$(@D)/python_include/pygetopt.h" && cp -f "/opt/python3.6/include/python3.6m/pyhash.h" "$(@D)/python_include/pyhash.h" && cp -f "/opt/python3.6/include/python3.6m/pylifecycle.h" "$(@D)/python_include/pylifecycle.h" && cp -f "/opt/python3.6/include/python3.6m/pymacconfig.h" "$(@D)/python_include/pymacconfig.h" && cp -f "/opt/python3.6/include/python3.6m/pymacro.h" "$(@D)/python_include/pymacro.h" && cp -f "/opt/python3.6/include/python3.6m/pymath.h" "$(@D)/python_include/pymath.h" && cp -f "/opt/python3.6/include/python3.6m/pymem.h" "$(@D)/python_include/pymem.h" && cp -f "/opt/python3.6/include/python3.6m/pyport.h" "$(@D)/python_include/pyport.h" && cp -f "/opt/python3.6/include/python3.6m/pystate.h" "$(@D)/python_include/pystate.h" && cp -f "/opt/python3.6/include/python3.6m/pystrcmp.h" "$(@D)/python_include/pystrcmp.h" && cp -f "/opt/python3.6/include/python3.6m/pystrhex.h" "$(@D)/python_include/pystrhex.h" && cp -f "/opt/python3.6/include/python3.6m/pystrtod.h" "$(@D)/python_include/pystrtod.h" && cp -f "/opt/python3.6/include/python3.6m/pythonrun.h" "$(@D)/python_include/pythonrun.h" && cp -f "/opt/python3.6/include/python3.6m/pythread.h" "$(@D)/python_include/pythread.h" && cp -f "/opt/python3.6/include/python3.6m/pytime.h" "$(@D)/python_include/pytime.h" && cp -f "/opt/python3.6/include/python3.6m/rangeobject.h" "$(@D)/python_include/rangeobject.h" && cp -f "/opt/python3.6/include/python3.6m/setobject.h" "$(@D)/python_include/setobject.h" && cp -f "/opt/python3.6/include/python3.6m/sliceobject.h" "$(@D)/python_include/sliceobject.h" && cp -f "/opt/python3.6/include/python3.6m/structmember.h" "$(@D)/python_include/structmember.h" && cp -f "/opt/python3.6/include/python3.6m/structseq.h" "$(@D)/python_include/structseq.h" && cp -f "/opt/python3.6/include/python3.6m/symtable.h" "$(@D)/python_include/symtable.h" && cp -f "/opt/python3.6/include/python3.6m/sysmodule.h" "$(@D)/python_include/sysmodule.h" && cp -f "/opt/python3.6/include/python3.6m/token.h" "$(@D)/python_include/token.h" && cp -f "/opt/python3.6/include/python3.6m/traceback.h" "$(@D)/python_include/traceback.h" && cp -f "/opt/python3.6/include/python3.6m/tupleobject.h" "$(@D)/python_include/tupleobject.h" && cp -f "/opt/python3.6/include/python3.6m/typeslots.h" "$(@D)/python_include/typeslots.h" && cp -f "/opt/python3.6/include/python3.6m/ucnhash.h" "$(@D)/python_include/ucnhash.h" && cp -f "/opt/python3.6/include/python3.6m/unicodeobject.h" "$(@D)/python_include/unicodeobject.h" && cp -f "/opt/python3.6/include/python3.6m/warnings.h" "$(@D)/python_include/warnings.h" && cp -f "/opt/python3.6/include/python3.6m/weakrefobject.h" "$(@D)/python_include/weakrefobject.h"
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
cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/__multiarray_api.h" "$(@D)/numpy_include/numpy/__multiarray_api.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/__ufunc_api.h" "$(@D)/numpy_include/numpy/__ufunc_api.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/_neighborhood_iterator_imp.h" "$(@D)/numpy_include/numpy/_neighborhood_iterator_imp.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/_numpyconfig.h" "$(@D)/numpy_include/numpy/_numpyconfig.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/arrayobject.h" "$(@D)/numpy_include/numpy/arrayobject.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/arrayscalars.h" "$(@D)/numpy_include/numpy/arrayscalars.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/halffloat.h" "$(@D)/numpy_include/numpy/halffloat.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/multiarray_api.txt" "$(@D)/numpy_include/numpy/multiarray_api.txt" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/ndarrayobject.h" "$(@D)/numpy_include/numpy/ndarrayobject.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/ndarraytypes.h" "$(@D)/numpy_include/numpy/ndarraytypes.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/noprefix.h" "$(@D)/numpy_include/numpy/noprefix.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h" "$(@D)/numpy_include/numpy/npy_1_7_deprecated_api.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/npy_3kcompat.h" "$(@D)/numpy_include/numpy/npy_3kcompat.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/npy_common.h" "$(@D)/numpy_include/numpy/npy_common.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/npy_cpu.h" "$(@D)/numpy_include/numpy/npy_cpu.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/npy_endian.h" "$(@D)/numpy_include/numpy/npy_endian.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/npy_interrupt.h" "$(@D)/numpy_include/numpy/npy_interrupt.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/npy_math.h" "$(@D)/numpy_include/numpy/npy_math.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/npy_no_deprecated_api.h" "$(@D)/numpy_include/numpy/npy_no_deprecated_api.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/npy_os.h" "$(@D)/numpy_include/numpy/npy_os.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/numpyconfig.h" "$(@D)/numpy_include/numpy/numpyconfig.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/old_defines.h" "$(@D)/numpy_include/numpy/old_defines.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/oldnumeric.h" "$(@D)/numpy_include/numpy/oldnumeric.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/ufunc_api.txt" "$(@D)/numpy_include/numpy/ufunc_api.txt" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/ufuncobject.h" "$(@D)/numpy_include/numpy/ufuncobject.h" && cp -f "/opt/python3.6/lib/python3.6/site-packages/numpy/core/include/numpy/utils.h" "$(@D)/numpy_include/numpy/utils.h"
   """,
)
