<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->

*   [Porting to a new platform](#porting-to-a-new-platform)
    *   [Requirements](#requirements)
    *   [Getting started](#getting-started)
    *   [Troubleshooting](#troubleshooting)
    *   [Optimizing for your platform](#optimizing-for-your-platform)
    *   [Code module organization](#code-module-organization)
    *   [Implementing more optimizations](#implementing-more-optimizations)

<!-- Added by: advaitjain, at: Mon 05 Oct 2020 02:36:46 PM PDT -->

<!--te-->

***Please note that we are currently pausing accepting new platforms***. Please
see our [contributions guide](../CONTRIBUTING.md) for more details and context.

Parts of the documentation below will likely change as we start accepting new
platform support again.

# Porting to a new platform

The remainder of this document provides guidance on porting TensorFlow Lite for
Microcontrollers to new platforms. You should read the
[developer documentation](https://www.tensorflow.org/lite/microcontrollers)
first.

## Requirements

Since the core neural network operations are pure arithmetic, and don't require
any I/O or other system-specific functionality, the code doesn't have to have
many dependencies. We've tried to enforce this, so that it's as easy as possible
to get TensorFlow Lite Micro running even on 'bare metal' systems without an OS.
Here are the core requirements that a platform needs to run the framework:

-   C/C++ compiler capable of C++11 compatibility. This is probably the most
    restrictive of the requirements, since C++11 is not as widely adopted in the
    embedded world as it is elsewhere. We made the decision to require it since
    one of the main goals of TFL Micro is to share as much code as possible with
    the wider TensorFlow codebase, and since that relies on C++11 features, we
    need compatibility to achieve it. We only use a small subset of C++ though,
    so don't worry about having to deal with template metaprogramming or
    similar challenges!

-   Debug logging. The core network operations don't need any I/O functions, but
    to be able to run tests and tell if they've worked as expected, the
    framework needs some way to write out a string to some kind of debug
    console. This will vary from system to system, for example on Linux it could
    just be `fprintf(stderr, debug_string)` whereas an embedded device might
    write the string out to a specified UART. As long as there's some mechanism
    for outputting debug strings, you should be able to use TFL Micro on that
    platform.

-   Math library. The C standard `libm.a` library is needed to handle some of
    the mathematical operations used to calculate neural network results.

-   Global variable initialization. We do use a pattern of relying on global
    variables being set before `main()` is run in some places, so you'll need to
    make sure your compiler toolchain supports this.

And that's it! You may be wondering about some other common requirements that
are needed by a lot of non-embedded software, so here's a brief list of things
that aren't necessary to get started with TFL Micro on a new platform:

-   Operating system. Since the only platform-specific function we need is
    `DebugLog()`, there's no requirement for any kind of Posix or similar
    functionality around files, processes, or threads.

-   C or C++ standard libraries. The framework tries to avoid relying on any
    standard library functions that require linker-time support. This includes
    things like string functions, but still allows us to use headers like
    `stdtypes.h` which typically just define constants and typedefs.
    Unfortunately this distinction isn't officially defined by any standard, so
    it's possible that different toolchains may decide to require linked code
    even for the subset we use, but in practice we've found it's usually a
    pretty obvious decision and stable over platforms and toolchains.

-   Dynamic memory allocation. All the TFL Micro code avoids dynamic memory
    allocation, instead relying on local variables on the stack in most cases,
    or global variables for a few situations. These are all fixed-size, which
    can mean some compile-time configuration to ensure there's enough space for
    particular networks, but does avoid any need for a heap and the
    implementation of `malloc\new` on a platform.

-   Floating point. Eight-bit integer arithmetic is enough for inference on many
    networks, so if a model sticks to these kind of quantized operations, no
    floating point instructions should be required or executed by the framework.

## Getting started

We recommend that you start trying to compile and run one of the simplest tests
in the framework as your first step. The full TensorFlow codebase can seem
overwhelming to work with at first, so instead you can begin with a collection
of self-contained project folders that only include the source files needed for
a particular test or executable. You can find a set of pre-generated projects
[here](https://drive.google.com/open?id=1cawEQAkqquK_SO4crReDYqf_v7yAwOY8).

As mentioned above, the one function you will need to implement for a completely
new platform is debug logging. If your device is just a variation on an existing
platform you may be able to reuse code that's already been written. To
understand what's available, begin with the default reference implementation at
[tensorflow/lite/micro/debug_log.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/debug_log.cc),
which uses fprintf and stderr. If your platform has this level of support for
the C standard library in its toolchain, then you can just reuse this.
Otherwise, you'll need to do some research into how your platform and device can
communicate logging statements to the outside world. As another example, take a
look at
[the Mbed version of `DebugLog()`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/mbed/debug_log.cc),
which creates a UART object and uses it to output strings to the host's console
if it's connected.

Begin by navigating to the micro_error_reporter_test folder in the pregenerated
projects you downloaded. Inside here, you'll see a set of folders containing all
the source code you need. If you look through them, you should find a total of
around 60 C or C++ files that compiled together will create the test executable.
There's an example makefile in the directory that lists all of the source files
and include paths for the headers. If you're building on a Linux or MacOS host
system, you may just be able to reuse that same makefile to cross-compile for
your system, as long as you swap out the `CC` and `CXX` variables from their
defaults, to point to your cross compiler instead (for example
`arm-none-eabi-gcc` or `riscv64-unknown-elf-gcc`). Otherwise, set up a project
in the build system you are using. It should hopefully be fairly
straightforward, since all of the source files in the folder need to be
compiled, so on many IDEs you can just drag the whole lot in. Then you need to
make sure that C++11 compatibility is turned on, and that the right include
paths (as mentioned in the makefile) have been added.

You'll see the default `DebugLog()` implementation in
'tensorflow/lite/micro/debug_log.cc' inside the micro_error_reporter_test
folder. Modify that file to add the right implementation for your platform, and
then you should be able to build the set of files into an executable. Transfer
that executable to your target device (for example by flashing it), and then try
running it. You should see output that looks something like this:

```
Number: 42
Badly-formed format string
Another  badly-formed  format string
~~ALL TESTS PASSED~~~
```

If not, you'll need to debug what went wrong, but hopefully with this small
starting project it should be manageable.

## Troubleshooting

When we've been porting to new platforms, it's often been hard to figure out
some of the fundamentals like linker settings and other toolchain setup flags.
If you are having trouble, see if you can find a simple example program for your
platform, like one that just blinks an LED. If you're able to build and run that
successfully, then start to swap in parts of the TF Lite Micro codebase to that
working project, taking it a step at a time and ensuring it's still working
after every change. For example, a first step might be to paste in your
`DebugLog()` implementation and call `DebugLog("Hello World!")` from the main
function.

Another common problem on embedded platforms is the stack size being too small.
Mbed defaults to 4KB for the main thread's stack, which is too small for most
models since TensorFlow Lite allocates buffers and other data structures that
require more memory. The exact size will depend on which model you're running,
but try increasing it if you are running into strange corruption issues that
might be related to stack overwriting.

## Optimizing for your platform

The default reference implementations in TensorFlow Lite Micro are written to be
portable and easy to understand, not fast, so you'll want to replace performance
critical parts of the code with versions specifically tailored to your
architecture. The framework has been designed with this in mind, and we hope the
combination of small modules and many tests makes it as straightforward as
possible to swap in your own code a piece at a time, ensuring you have a working
version at every step. To write specialized implementations for a platform, it's
useful to understand how optional components are handled inside the build
system.

## Code module organization

We have adopted a system of small modules with platform-specific implementations
to help with portability. Every module is just a standard `.h` header file
containing the interface (either functions or a class), with an accompanying
reference implementation in a `.cc` with the same name. The source file
implements all of the code that's declared in the header. If you have a
specialized implementation, you can create a folder in the same directory as the
header and reference source, name it after your platform, and put your
implementation in a `.cc` file inside that folder. We've already seen one
example of this, where the Mbed and Bluepill versions of `DebugLog()` are inside
[mbed](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/mbed)
and
[bluepill](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/bluepill)
folders, children of the
[same directory](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite//micro)
where the stdio-based
[`debug_log.cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/debug_log.cc)
reference implementation is found.

The advantage of this approach is that we can automatically pick specialized
implementations based on the current build target, without having to manually
edit build files for every new platform. It allows incremental optimizations
from a always-working foundation, without cluttering the reference
implementations with a lot of variants.

To see why we're doing this, it's worth looking at the alternatives. TensorFlow
Lite has traditionally used preprocessor macros to separate out some
platform-specific code within particular files, for example:

```
#ifndef USE_NEON
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#include <arm_neon.h>
#endif
```

There’s also a tradition in gemmlowp of using file suffixes to indicate
platform-specific versions of particular headers, with kernel_neon.h being
included by kernel.h if `USE_NEON` is defined. As a third variation, kernels are
separated out using a directory structure, with
tensorflow/lite/kernels/internal/reference containing portable implementations,
and tensorflow/lite/kernels/internal/optimized holding versions optimized for
NEON on Arm platforms.

These approaches are hard to extend to multiple platforms. Using macros means
that platform-specific code is scattered throughout files in a hard-to-find way,
and can make following the control flow difficult since you need to understand
the macro state to trace it. For example, I temporarily introduced a bug that
disabled NEON optimizations for some kernels when I removed
tensorflow/lite/kernels/internal/common.h from their includes, without realizing
it was where USE_NEON was defined!

It’s also tough to port to different build systems, since figuring out the right
combination of macros to use can be hard, especially since some of them are
automatically defined by the compiler, and others are only set by build scripts,
often across multiple rules.

The approach we are using extends the file system approach that we use for
kernel implementations, but with some specific conventions:

-   For each module in TensorFlow Lite, there will be a parent directory that
    contains tests, interface headers used by other modules, and portable
    implementations of each part.
-   Portable means that the code doesn’t include code from any libraries except
    flatbuffers, or other TF Lite modules. You can include a limited subset of
    standard C or C++ headers, but you can’t use any functions that require
    linking against those libraries, including fprintf, etc. You can link
    against functions in the standard math library, in <math.h>.
-   Specialized implementations are held inside subfolders of the parent
    directory, named after the platform or library that they depend on. So, for
    example if you had my_module/foo.cc, a version that used RISC-V extensions
    would live in my_module/riscv/foo.cc. If you had a version that used the
    CMSIS library, it should be in my_module/cmsis/foo.cc.
-   These specialized implementations should completely replace the top-level
    implementations. If this involves too much code duplication, the top-level
    implementation should be split into smaller files, so only the
    platform-specific code needs to be replaced.
-   There is a convention about how build systems pick the right implementation
    file. There will be an ordered list of 'tags' defining the preferred
    implementations, and to generate the right list of source files, each module
    will be examined in turn. If a subfolder with a tag’s name contains a .cc
    file with the same base name as one in the parent folder, then it will
    replace the parent folder’s version in the list of build files. If there are
    multiple subfolders with matching tags and file names, then the tag that’s
    latest in the ordered list will be chosen. This allows us to express “I’d
    like generically-optimized fixed point if it’s available, but I’d prefer
    something using the CMSIS library” using the list 'fixed_point cmsis'. These
    tags are passed in as `TAGS="<foo>"` on the command line when you use the
    main Makefile to build.
-   There is an implicit “reference” tag at the start of every list, so that
    it’s possible to support directory structures like the current
    tensorflow/kernels/internal where portable implementations are held in a
    “reference” folder that’s a sibling to the NEON-optimized folder.
-   The headers for each unit in a module should remain platform-agnostic, and
    be the same for all implementations. Private headers inside a sub-folder can
    be used as needed, but shouldn’t be referred to by any portable code at the
    top level.
-   Tests should be at the parent level, with no platform-specific code.
-   No platform-specific macros or #ifdef’s should be used in any portable code.

The implementation of these rules is handled inside the Makefile, with a
[`specialize` function](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/tools/make/helper_functions.inc#L42)
that takes a list of reference source file paths as an input, and returns the
equivalent list with specialized versions of those files swapped in if they
exist.

## Implementing more optimizations

Clearly, getting debug logging support is only the beginning of the work you'll
need to do on a particular platform. It's very likely that you'll want to
optimize the core deep learning operations that take up the most time when
running models you care about. The good news is that the process for providing
optimized implementations is the same as the one you just went through to
provide your own logging. You'll need to identify parts of the code that are
bottlenecks, and then add specialized implementations in their own folders.
These don't need to be platform specific, they can also be broken out by which
library they rely on for example. [Here's where we do that for the CMSIS
implementation of integer fast-fourier
transforms](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_speech/simple_features/simple_features_generator.cc).
This more complex case shows that you can also add helper source files alongside
the main implementation, as long as you
[mention them in the platform-specific makefile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_speech/CMSIS/Makefile.inc).
You can also do things like update the list of libraries that need to be linked
in, or add include paths to required headers.
