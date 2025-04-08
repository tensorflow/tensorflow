f2reduce: a MIT-licenced library for Gaussian elimination over GF(2)
====================================================================

This is a very lightweight implementation for converting a binary matrix
to row reduced echelon form. It incorporates the following optimisations:

 - Kronrod's algorithm ('method of four Russians');
 - Designed to properly autovectorise in both GCC and LLVM;
 - Attempts to ensure that memory loads/stores are cache-aligned;
 - Designed to achieve high instruction-level parallelism;
 - Able to use AVX512's `vpternlogq` instruction if present;
 - Minimal memory overhead (a few megabytes).

There are no architecture-specific intrinsics or assembly, so this should
work well on any architecture where the compiler can autovectorise.

For simplicity, we do not use Strassen, so our performance is overtaken by
[M4RI][1] whenever the matrices are large and have full column rank.

For all other cases, we have several advantages over M4RI:

 - Substantially better performance on small, wide, or low-rank matrices;
 - MIT-licenced rather than GPL-licenced;
 - No assumptions about the processor architecture;
 - No configuration required (`-O3 -march=native` is enough).

We expose a single function with the following signature:

    void inplace_rref_strided(uint64_t *matrix, uint64_t rows, uint64_t cols, uint64_t stride);

The matrix should be in row-major format and is overwritten in-place. The
`stride` parameter specifies the offset between adjacent rows **in 64-bit
words, not bytes**. The mapping between matrix entries and memory is as
follows:

    the (j+64*k)th entry of the ith row is (matrix[i * stride + k] >> j) & 1

Since the performance can depend on the stride and how it interacts with
processor caches, we expose another function to return a recommended stride:

    uint64_t get_recommended_stride(uint64_t cols);

Although `f2reduce` is compiled in C++11, the resulting static library
has C-linkage so can be called from any C/C++ code.

Dependencies
------------

`f2reduce` has no dependencies; just compile `f2reduce.cpp` with the
`-O3 -march=native` flags to produce a static library and include the header
file `f2reduce.h` in your project.

The automated test suite has dependencies on [M4RI][1] (for benchmarking
timings against M4RI and checking that implementations agree), [GoogleTest][2]
(for unit testing), and [cpads][3] (for high-quality pseudo-random number
generation). Downloading of the dependencies and building of the test suite
is automated by [CMake][4].

To build the test suite, you need to manually append `add_subdirectory(test)`
to the end of the `CMakeLists.txt` file. This is so that `f2reduce` does not
have any build dependencies by default.

[1]: https://github.com/malb/m4ri
[2]: https://github.com/google/googletest
[3]: https://gitlab.com/hatsya/open-source/cpads
[4]: https://cmake.org/
