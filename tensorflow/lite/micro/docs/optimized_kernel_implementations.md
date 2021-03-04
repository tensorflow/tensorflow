<!-- mdformateoff(b/169948621#comment2) -->

<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->

*   [Summary](#summary)
*   [High-Level Steps](#high-level-steps)
    *   [Why not Optimize the Reference Kernels](#why-not-optimize-the-reference-kernels)
*   [Software Architecture](#software-architecture)
    *   [Hardware-specific NN library](#hardware-specific-nn-library)
    *   [Optimized Kernels](#optimized-kernels)
    *   [Build System Integration](#build-system-integration)
    *   [Testing and Continuous Integration](#testing-and-continuous-integration)

<!-- Added by: advaitjain, at: Wed 17 Feb 2021 02:14:16 PM PST -->

<!--te-->

# Summary

This guide describes the recommended high-level architecture and steps to add
hardware-specific optimized kernels to TfLite Micro.

The goal with these optimizations and the process that we recommend to getting
them merged into the TfLite Micro codebase is to have a measurable and
documented performance improvement on a benchmark of interest.

Once the optimizations are merged, they will indeed be used for more than the
benchmark but the context for why the optimizations were added is still very
important.

# High-Level Steps

1.  Pick a benchmark that you would like to measure the performance for.

    *   Existing benchmarks are in the [benchmarks directory](../benchmarks).
    *   If none of the existing benchmarks capture your use-case, then please
        create a github issue or start a thread on micro@tensorflow.org to
        figure out how to add in a new benchmark.
    *   If adding a publicly-available benchmark to the TFLM codebase is
        determined to be infeasible, then a fall-back would be to have an
        internal benchmark that can be used to document the benefits of adding
        in the optimizations via PR descriptions.
    *   Adding optimized code without any associated benchmarks will need very
        strong justification and will most likely not be permitted.

1.  Do the groundwork and architecture needed to be able to add in optimizations
    for your target (more details in the
    [software architecture](#software-architecture) section).

1.  Create one pull request for each optimized kernel with the PR description
    clearly stating the commands that were used to measure the performance
    improvement.

    *   This context is important even if the toolchain is proprietary and there
        are currently a small number of users.
        *   See [this PR](https://github.com/tensorflow/tensorflow/pull/47098)
            as an example.
        *   At minimum the latency with and without the particular optimized
            kernel should be documented.
            [Additional context](https://github.com/tensorflow/tensorflow/pull/46746)
            may also be desirable.
    *   Here is some
        [general guidance](https://testing.googleblog.com/2017/09/code-health-providing-context-with.html)
        on writing
        [good PR descriptions](https://google.github.io/eng-practices/review/developer/cl-descriptions.html)

## Why Not Optimize the Portable Reference Kernels?

We would like to explicitly point out (as have others) that the reference kernel
implementations are not performant and there are plenty of opportunities to
speed them up. This is by design and the reference kernels are meant to be a
shared starting point to then be optimized in a target specific optimized kernel
implementation.

Two previous discussions on this topic are on
[PR #42477](https://github.com/tensorflow/tensorflow/pull/42477) and
[PR #45227](https://github.com/tensorflow/tensorflow/pull/45227)

Our current point of view on this topic is that while optimizing shared
reference code in a portable manner is attractive, we are making an explicit
choice to not go down that path and instead rely on target-specific optimized
implementations. The TFLM codebase has a growing list of optimized kernel
implementations, and we are investing in making the process of adding new
implementations smoother.

# Software Architecture

The optimized kernel architecture is composed of the following three modules:

1.  Hardware-specific NN library
1.  Optimized Kernels
1.  Build System Integration

## Hardware-specific NN library

This library uses knowledge of the hardware and compiler to implement the
underlying operations. Examples of this are
[CMSIS-NN](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN) from
ARM and [NNLib](https://github.com/foss-xtensa/nnlib-hifi4) from Cadence.

The benefits of having this API separation are:

1.  The NN library does not need to follow the style guide of the rest of the
    TFLM code.
1.  Releases of the NN library can be made independent of TFLM
1.  The same NN library can be used and tested independent of TFLM.
1.  The maintainers of the NN library have full control over the development
    process that they would like to follow.

## Optimized Kernels

These will be (hopefully thin) wrappers that act as the glue between TFLM and
the NN library.

The goal here is to delegate as much work as possible to the NN library while
still allowing the two APIs (TFLM and NN library) to be independent of each
other. If there is a performance degradation due to this (for example,
unnecessary memory copies) then we can evaluate those on a case-by-case basis.

This code will be reviewed and merged in the TFLM github repository and must
follow the development style of the TFLM codebase.

Some amount of refactoring of the existing code may be needed to ensure that
code is suitably shared between the reference and optimized kernels. There is
currently no fixed recipe for this refactor and we will evaluate on a
case-by-case basis during the PR review.

For example, to add an optimized implementation for `fully_conntected` for the
Xtensa Fusion F1 the steps were: *
[PR 1](https://github.com/tensorflow/tensorflow/pull/45464): refactor for
reference fallbacks and a baseline latency. *
[PR 2](https://github.com/tensorflow/tensorflow/pull/46242): refactor to share
code between reference and optimized kernels. *
[PR 3](https://github.com/tensorflow/tensorflow/pull/46411): add the code needed
to use the optimized NN lib and document the latency improvement.

## Build System Integration

This module is the least defined but we strongly recommend the following: 1. A
single target makefile.inc for all the architectures that you would like to
support along with optional target-specific
[system_setup.cc](../arduino/system_setup.cc). See
[cortex_m_generic_makefile.inc](../tools/make/targets/cortex_m_generic_makefile.inc)
and [xtensa_makefile.inc](../tools/make/targets/xtensa_makefile.inc) as
examples.

1.  A single `ext_libs.inc` (and associated scripts) that downloads any external
    dependencies (including the NN library). For example:

    *   [cmsis_nn.inc](../tools/make/ext_libs/cmsis_nn.inc) and
        [cmsis_download.sh](../tools/make/ext_libs/cmsis_download.sh)
    *   [xtensa.inc](../tools/make/ext_libs/xtensa.inc) and
        [xtensa_download.sh](../tools/make/ext_libs/xtensa_download.sh)

1.  The optimized kernels will then live in a kernels subdirectory (e.g.
    [kernels/cmsis_nn](../kernels/cmsis_nn) and
    [kernels/xtensa](../kernels/xtensa))

Two development workflows that the TFLM team would like to encourage and
support:

1.  Export static library + headers into target-specific development environment

    *   Build a static libtensorflow-microlite.a using the TFLM makefile with:
        `make -f tensorflow/lite/micro/tools/make/Makefile TARGET=<target>
        OPTIMIZED_KERNEL_DIR=<optimize_dir> microlite`
    *   Use the static library and any TFLM headers as part of the overall
        application (with its own build system).

1.  Integrate TFLM with IDE:

    *   This has historically been done using the TFLM Makefile’s support for
        project generation.

    *   However, given the learning curve and high-maintenance overhead, we are
        moving away from supporting project generation via the Makefile and are
        encouraging future IDE integrations to be done outside of the TFLM
        Makefiles.

    *   The TFLM team is currently working through the details on this topic.

## Testing and Continuous Integration

The kernel tests are the primary method of ensuring that the optimized kernel
implementations are accurate.

Currently, most of the tests require the optimizations to be bit-exact to the
quantized reference implementation. We can revisit this requirement if it ends
up having a high associated cost on the latency.

We strongly encourage optimized kernel implementations to have an associated
continuous build that runs through all the unit tests and publishes a build
badge to the
[TFLM community supported builds](../README.md#community-supported-builds)
table. Running the units tests once a day is often a good place to start.
