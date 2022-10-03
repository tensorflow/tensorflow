# XLA Runtime

XLA runtime is a set of libraries that support execution of XLA programs
compiled to native executables. XLA runtime provides user-friendly APIs for
calling compiled programs, takes care of passing arguments and returning
results according to the expected ABI, implements async tasks support and
defines the FFI for compiled programs to call into user-defined callbacks.

If you squint and look at XLA as a programming language like Objective-C, then
the XLA runtime is somewhat similar to Objective-C runtime: a runtime library
that provides support for the functionality that we do not want to compile, e.g.
it provides functionality to launch asynchronous tasks in a thread pool, because
we do not want to codegen directly on top of `pthreads` library.
