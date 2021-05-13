<!-- mdformat off(b/169948621#comment2) -->

<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->
   * [Profiling](#profiling)
      * [API](#api)
      * [Per-Op Profiling](#per-op-profiling)
      * [Subroutine Profiling](#subroutine-profiling)

<!-- Added by: njeff, at: Wed 04 Nov 2020 04:35:07 PM PST -->

<!--te-->

# Profiling

This doc outlines how to use the TFLite Micro profiler to gather information
about per-op invoke duration and to use the profiler to identify bottlenecks
from within operator kernels and other TFLite Micro routines.

## API

The MicroInterpreter class constructor contains and optional profiler argument.
This profiler must be an instance of the tflite::Profiler class, and should
implement the BeginEvent and EndEvent methods. There is a default implementation
in tensorflow/lite/micro/micro_profiler.cc which can be used for most purposes.

## Per-Op Profiling

There is a feature in the MicroInterpreter to enable per-op profiling. To enable
this, provide a MicroProfiler to the MicroInterpreter's constructor then build
with a non-release build to disable the NDEBUG define surrounding the
ScopedOperatorProfile within the MicroInterpreter.

## Subroutine Profiling

In order to further dig into performance of specific routines, the MicroProfiler
can be used directly from the TFLiteContext or a new MicroProfiler can be
created if the TFLiteContext is not available where the profiling needs to
happen. The MicroProfiler's BeginEvent and EndEvent can be called directly, or
wrapped using a [ScopedProfile](../../lite/core/api/profiler.h).
