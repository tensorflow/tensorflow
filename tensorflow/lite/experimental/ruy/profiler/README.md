# A minimalistic profiler sampling pseudo-stacks

## Overview

The present directory is the "ruy profiler". As a time profiler, it allows to
measure where code is spending time.

Contrary to most typical profilers, what it samples is not real call stacks, but
"pseudo-stacks" which are just simple data structures constructed from within
the program being profiled. Using this profiler requires manually instrumenting
code to construct such pseudo-stack information.

Another unusual characteristic of this profiler is that it uses only the C++11
standard library. It does not use any non-portable feature, in particular it
does not rely on signal handlers. The sampling is performed by a thread, the
"profiler thread".

A discussion of pros/cons of this approach is appended below.

## How to use this profiler

### How to instrument code

An example of instrumented code is given in `test_instrumented_library.cc`.

Code is instrumented by constructing `ScopeLabel` objects. These are RAII
helpers, ensuring that the thread pseudo-stack contains the label during their
lifetime. In the most common use case, one would construct such an object at the
start of a function, so that its scope is the function scope and it allows to
measure how much time is spent in this function.

```c++
#include "ruy/profiler/instrumentation.h"

...

void SomeFunction() {
  ruy::profiling::ScopeLabel function_label("SomeFunction");
  ... do something ...
}
```

A `ScopeLabel` may however have any scope, for instance:

```c++
if (some_case) {
  ruy::profiling::ScopeLabel extra_work_label("Some more work");
  ... do some more work ...
}
```

The string passed to the `ScopeLabel` constructor must be just a pointer to a
literal string (a `char*` pointer). The profiler will assume that these pointers
stay valid until the profile is finalized.

However, that literal string may be a `printf` format string, and labels may
have up to 4 parameters, of type `int`. For example:

```c++
void SomeFunction(int size) {
  ruy::profiling::ScopeLabel function_label("SomeFunction (size=%d)", size);

```

### How to run the profiler

Profiling instrumentation is a no-op unless the preprocessor token
`RUY_PROFILER` is defined, so defining it is the first step when actually
profiling. When building with Bazel, the preferred way to enable that is to pass
this flag on the Bazel command line:

```
--define=ruy_profiler=true
```

To actually profile a code scope, it is enough to construct a `ScopeProfile`
object, also a RAII helper. It will start the profiler on construction, and on
destruction it will terminate the profiler and report the profile treeview on
standard output by default. Example:

```c++
void SomeProfiledBenchmark() {
  ruy::profiling::ScopeProfile profile;

  CallSomeInstrumentedCode();
}
```

An example is provided by the `:test` target in the present directory. Run it
with `--define=ruy_profiler=true` as explained above:

```
bazel run -c opt \
   --define=ruy_profiler=true \
  //tensorflow/lite/experimental/ruy/profiler:test
```

The default behavior dumping the treeview on standard output may be overridden
by passing a pointer to a `TreeView` object to the `ScopeProfile` constructor.
This causes the tree-view to be stored in that `TreeView` object, where it may
be accessed an manipulated using the functions declared in `treeview.h`. The
aforementioned `:test` provides examples for doing so.

## Advantages and inconvenients

Compared to a traditional profiler, e.g. Linux's "perf", the present kind of
profiler has the following inconvenients:

*   Requires manual instrumentation of code being profiled.
*   Substantial overhead, modifying the performance characteristics of the code
    being measured.
*   Questionable accuracy.

But also the following advantages:

*   Profiling can be driven from within a benchmark program, allowing the entire
    profiling procedure to be a single command line.
*   Not relying on symbol information removes removes exposure to toolchain
    details and means less hassle in some build environments, especially
    embedded/mobile (single command line to run and profile, no symbols files
    required).
*   Fully portable (all of this is standard C++11).
*   Fully testable (see `:test`). Profiling becomes just another feature of the
    code like any other.
*   Customized instrumentation can result in easier to read treeviews (only
    relevant functions, and custom labels may be more readable than function
    names).
*   Parametrized/formatted labels allow to do things that aren't possible with
    call-stack-sampling profilers. For example, break down a profile where much
    time is being spent in matrix multiplications, by the various matrix
    multiplication shapes involved.

The philosophy underlying this profiler is that software performance depends on
software engineers profiling often, and a key factor limiting that in practice
is the difficulty or cumbersome aspects of profiling with more serious profilers
such as Linux's "perf", espectially in embedded/mobile development: multiple
command lines are involved to copy symbol files to devices, retrieve profile
data from the device, etc. In that context, it is useful to make profiling as
easy as benchmarking, even on embedded targets, even if the price to pay for
that is lower accuracy, higher overhead, and some intrusive instrumentation
requirement.

Another key aspect determining what profiling approach is suitable for a given
context, is whether one already has a-priori knowledge of where much of the time
is likely being spent. When one has such a-priori knowledge, it is feasible to
instrument the known possibly-critical code as per the present approach. On the
other hand, in situations where one doesn't have such a-priori knowledge, a real
profiler such as Linux's "perf" allows to right away get a profile of real
stacks, from just symbol information generated by the toolchain.
