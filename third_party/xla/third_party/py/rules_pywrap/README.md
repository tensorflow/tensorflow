# Pywrap Rules

## Overview
The purpose of Pywrap rules is to automate creation of Python bindings (calling
C++ code from Python) across bazel projects. The rules were designed to address
the following common issues when dealing with Python bindings in bazel
projects.

### Prevent ODR Violations
If the common portion between two or more modules is not handled carefully it
may lead to various difficult to debug errors which are often referenced to
as [ODR violations](https://en.cppreference.com/w/cpp/language/definition.html).

Preventing ODRs is usually the main driving factor for using Pywrap rules, as
they were specifically designed for cases when there are two or more Python
modules implemented in C++ which share common code (a very common case for any
sufficiently big project and a main driving force for most ODRs).

### Simplify Cross-Platform Development
Pywrap rules provide uniform interface for cross-platform development, meaning
the rules surface looks the same regardless if it is built for Linux, Mac or
Windows, shielding the user from the need to know nitty-gritty details of
dynamic linking model on each of those platforms. It also provides a unified
interface between Google's internal and OSS builds.

### Minimize Binary Sizes
In many cases using Pywrap rules may lead to smaller overall artifact sizes, by
preventing common positions of the code being uncontrollably duplicated across
different binaries.

### Simplify Binary Packaging
Pywrap rules also provide functionality needed for proper packaging of built
binaries in Python wheels. The rules do not handle packaging in .whl files
directly and rely on external code to do it, but they make sure the different
binaries which were built and tested together in bazel, are still going to work
together the exact same way when put in an independent Python wheel regardless
of the target platform (Linux, Mac or Windows).

## Rules API
There are only two main rules you will be using most of the time:

- `python_extension` - builds each individual Python binding.
- `pywrap_library` - links python bindings produced by `python_extension` into a
  set of specific dynamic libraries, ready to be used by tests, binaries and
  packaging rules.

There is also a thin wrapper on top of `python_extension` called
`pybind_extension`, which simply automatically adds dependency on `pybind11` to
`python_extension`. With that being said, for the rest of this document and in
rules API documentation, `python_extension` and `pybind_extension` could be used
interchangeably.

There are a few more public macros, which are used to directly access produced
binary artifacts:

- `pywrap_binaries` - returns all binaries produced by `pywrap_library`
- `pywrap_common_library` - returns a specific common library, among multiple
  ones produced by `pywrap_library`

### Quick start
For detailed information please refer to each individual rule API documentation
in `pywrap.impl.bzl`.

Use `cc_library` or other rules returning `CcInfo` provider to compile all the
code you need for your Python bindings as their potentially common dependencies.

Call `python_extension` for each of the bindings, providing all necessary
`cc_library` in `deps` parameter, pass the code that does actual python bindings
definitions (e.g. calls to pybind11 macros) as its `srcs` parameter.

The name of the `python_extension` target will be the name of your new Python
module, the bazel package in which the target was declared will become this
module's fully qualified name.

For example, if you have `python_extension(name = "your_awesome_module")`
defined in `their/awesome/package/BUILD`, use
`from their.awesome.package import your_awesome_module` syntax to import it in
Python code.

Any `python_extension` can be used as a dependency of any `py_library` target in
your project that imports `my_awesome_module`. While it can also be passed as a
dependency to any of your `py_test` or `py_binary` targets, doing so will have
no effect, as to become runnable (i.e. to be linked in an actual os-specific
binary) `python_extension` targets need to be linked together by
`pywrap_library` rule first.

Note, each `pybind_extension` can be compiled independently, but to work
together without ODR violations and other nasty errors they must be linked
together within one target, because only when all bindings are present together
it is possible to determine what is used by what and link all the stuff properly
without duplicates - `pywrap_library` does exactly that.

A rule of thumb: a project is expected to have multiple `python_extension`
targets, one for each python bindings module, but have very little, in most
cases only one `pywrap_library` target.

Once you have all of your `python_extension` targets defined, determine which of
the modules are supposed to work together and put all of those into a single
`pywrap_library`.

Here is the part that often confuses users: each of your `python_extension`
targets may be in different packages which may be different (and likely will)
from the one where you put the corresponding `pywrap_library`. In that case, for
dynamic libraries to be able to find each other during runtime, you need to pass
an extra `common_lib_packages` parameter to each of your failing
`python_extension` targets. For example if your `pywrap_library` target is in a
package `some/other/package`, add `common_lib_packages = ["some/other/package"]`
to each of your python extension targets that belong to different packages. If
you receive a "Could not import original test/binary location" error during
runtime, it is most likely because of this package mismatch between
`pythong_extension` and `pywrap_library`.

Depend on your `pywrap_library` in any of your `py_test` or `py_binary` targets,
or any other targets which know what to do with dynamic libraries (assuming you
know what you are doing).

### Dependency Management Cheat Sheet
People often get confused by what should depend on what. Use the following
cheatsheet to validate that you structured your project properly.

#### Dependency Do's
Your project is well-structured if:

- `python_extension` targets depend on `cc_library`;
- `py_library` targets depend on `python_extension`;
- `pywrap_library` targets depend on multiple `python_extension` either directly
  or transitively (via `py_library`);
- `py_test` and `py_binary` targets depend on `pywrap_library`;
- if something depends on `pywrap_library` it depends only on one
  `pywrap_library` in its entire transitive closure.

#### Dependency Don'ts
Please make sure your project does not have the following:

- `pywrap_library` targets should not depend on each other; while it is
  technically a valid case, it is valid only if none of `pywrap_libraries` have
  common pieces (i.e. shared CcInfo provider anywhere in their transitive
  closure);
- `py_test` and `py_binary` depending on `python_extension` directly or
  transitively will have no effect, depend on corresponding `pywrap_library`
  instead;
- although `py_library` may depend on `pywrap_library`, there should be no
  reason to do so, and there still must be only one `pywrap_library` in a
  transitive closure of any `py_library`, which will be very hard to control, if
  you start making this dependency.

### Example
A typical project structure should be as follows:

```python
cc_library(
    name = "second_library",
    hdrs = ["second_library.h"],
    srcs = ["second_library.cc"],
)
cc_library(
    name = "first_library",
    hdrs = ["first_library.h"],
    srcs = ["first_library.cc"],
    deps = [":second_library"],
)
pybind_extension(
    name = "pybind",
    srcs = ["pybind.cc"],
    deps = [":first_library", ":second_library"],
    # Set if pywrap_library is in different package
    common_lib_packages = [],
    # Set to True if you wan ultra-thin python bindings (recommended)
    wrap_py_init = False,
)
pybind_extension(
    name = "pybind_copy",
    srcs = ["pybind_copy.cc"],
    deps = [":first_library", ":second_library"],
    # Set if pywrap_library is in different package
    common_lib_packages = [],
    # Set to True if you wan ultra-thin python bindings (recommended)
    wrap_py_init = False,
)
py_library(
    name = "optional_py_library",
    # You may pass any of these directly to pywrap_library instead
    deps = [":pybind", ":pybind_copy"]
)
pywrap_library(
    name = "pybind_aggregated",
    deps = [":optional_py_library"],
)
py_test(
    name = "pybind_py_test",
    srcs = ["pybind_py_test.py"],
    deps = [":pybind_aggregated"],
)
```

### Misc
There are many more advance features in Pywrap libraries which exist to
facilitate various non-core workflows, such as backward compatibility for
projects which were built without pywrap originally, platform-specific
features (like .def files for windows) etc.

Pleaser refer to each individual rule documentation for details how to use that.
Here are some hints to steer you in the right direction:

- if you need to control how common (libraries which contain code shared by
  different python bindings) libraries behave (like which symbols they export or
  how they are named in the final artifact) use `common_lib_`-prefixed
  parameters to `pywrap_library` rule.
- if you need to filter out test-only code from your final artifacts look into
  `starlark_only` arguments in both `python_extension` and `pywrap_library`.
- If you want a super-thin python binding `.so`/`.pyd` object (the ones which
  will contain only `PyInit_<module_name>` symbol in them, and the rest will be
  in the common lib) add `wrap_py_init = True` argument to your
  `python_extension` targets. This is especially recommended for Windows builds.
