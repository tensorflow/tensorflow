#include "f2reduce.h"

#include <string.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#define RESTRICT __restrict
#define NO_INLINE __declspec(noinline)
#elif defined(__GNUC__)
#define RESTRICT __restrict__
#define NO_INLINE __attribute__ ((noinline))
#endif

namespace f2reduce {

static void swap_rows(uint64_t* RESTRICT x, uint64_t* RESTRICT y, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        uint64_t z = x[i]; x[i] = y[i]; y[i] = z;
    }
}

// the noinline attribute is necessary for gcc to properly vectorise this:
template<uint64_t N>
static NO_INLINE void memxor_lop7(uint64_t* RESTRICT dst,
    const uint64_t* RESTRICT src1,
    const uint64_t* RESTRICT src2,
    const uint64_t* RESTRICT src3,
    const uint64_t* RESTRICT src4,
    const uint64_t* RESTRICT src5,
    const uint64_t* RESTRICT src6) {
    for (uint64_t i = 0; i < N; i++) {
        dst[i] ^= src1[i] ^ src2[i] ^ src3[i] ^ src4[i] ^ src5[i] ^ src6[i];
    }
}

template<uint64_t N>
static NO_INLINE void memxor_lop5(uint64_t* RESTRICT dst,
    const uint64_t* RESTRICT src1,
    const uint64_t* RESTRICT src2,
    const uint64_t* RESTRICT src3,
    const uint64_t* RESTRICT src4) {
    for (uint64_t i = 0; i < N; i++) {
        dst[i] ^= src1[i] ^ src2[i] ^ src3[i] ^ src4[i];
    }
}

template<uint64_t N>
static NO_INLINE void memxor_lop3(uint64_t* RESTRICT dst,
    const uint64_t* RESTRICT src1,
    const uint64_t* RESTRICT src2) {
    for (uint64_t i = 0; i < N; i++) {
        dst[i] ^= src1[i] ^ src2[i];
    }
}

template<uint64_t N>
static void memxor_inplace(uint64_t* RESTRICT dst, const uint64_t* RESTRICT src1, const uint64_t* RESTRICT src2) {
    for (uint64_t i = 0; i < N; i++) {
        dst[i] = src1[i] ^ src2[i];
    }
}

// split k into 6 approximately-equal pieces
static void split_k(int k, int* subkays) {

    int k5_k6 = (k <= 32) ? 0 : (k / 3);
    int k3_k4 = (k - k5_k6) >> 1;
    int k1_k2 = k - k5_k6 - k3_k4;

    subkays[0] = k1_k2 >> 1;
    subkays[1] = k1_k2 - subkays[0];
    subkays[2] = k3_k4 >> 1;
    subkays[3] = k3_k4 - subkays[2];
    subkays[4] = k5_k6 >> 1;
    subkays[5] = k5_k6 - subkays[4];
}



/**
 * Sextuple Kronrod implementation.
 *
 * This populates six lookup tables of approximately-equal sizes where each
 * entry (8*N bytes) contains a linear combination of rows. The transformation
 * encoded in 'workspace' is then applied using ternary XORs which are very
 * AVX512-friendly.
 */
template<uint64_t N>
static void kronrod(uint64_t* RESTRICT matrix, uint64_t rows, uint64_t stride, const uint64_t* RESTRICT workspace, uint64_t* RESTRICT cache, const uint64_t* RESTRICT pivots, int k) {
    constexpr int logwidth = 5;

    static_assert(N <= (1ull << logwidth), "kronrod<N> assumes that N <= 32");

    int subkays[6];
    int cumkays[6];
    uint64_t* caches[6];
    split_k(k, subkays);

    caches[0] = cache;
    cumkays[0] = 0;

    for (int i = 0; i < 5; i++) {
        caches[i+1] = caches[i] + (1ull << (subkays[i] + logwidth));
        cumkays[i+1] = cumkays[i] + subkays[i];
    }

    // build:
    for (int o = 0; o < 6; o++) {
        uint64_t* subcache = caches[o];
        memset(subcache, 0, 8 << logwidth);
        for (int j = 0; j < subkays[o]; j++) {
            uint64_t p = (1ull << j);
            memcpy(subcache + (p << logwidth), matrix + pivots[j + cumkays[o]] * stride, N * 8);
            for (uint64_t i = 1; i < p; i++) {
                memxor_inplace<N>(subcache + ((i+p) << logwidth), subcache + (i << logwidth), subcache + (p << logwidth));
            }
        }
    }

    uint64_t mask0 = (1ull << subkays[0]) - 1;
    uint64_t mask1 = (1ull << subkays[1]) - 1;
    uint64_t mask2 = (1ull << subkays[2]) - 1;
    uint64_t mask3 = (1ull << subkays[3]) - 1;
    uint64_t mask4 = (1ull << subkays[4]) - 1;
    uint64_t mask5 = (1ull << subkays[5]) - 1;

    // apply:
    for (uint64_t r = 0; r < rows; r++) {

        if (N >= 32) {
            // prefetch 256 bytes, 15 rows later:
            uint64_t* ppp = matrix + (r + 15) * stride;
#if defined(__GNUC__)
            __builtin_prefetch(ppp);
            __builtin_prefetch(ppp + 8);
            __builtin_prefetch(ppp + 16);
            __builtin_prefetch(ppp + 24);
#endif
        }

        uint64_t w = workspace[r];

        uint64_t w0 = w & mask0;
        uint64_t w1 = (w >> cumkays[1]) & mask1;
        uint64_t w2 = (w >> cumkays[2]) & mask2;
        uint64_t w3 = (w >> cumkays[3]) & mask3;
        if (k <= 32) {
            memxor_lop5<N>(matrix + r * stride,
                            caches[0] + (w0 << logwidth),
                            caches[1] + (w1 << logwidth),
                            caches[2] + (w2 << logwidth),
                            caches[3] + (w3 << logwidth));
        } else {
            uint64_t w4 = (w >> cumkays[4]) & mask4;
            uint64_t w5 = (w >> cumkays[5]) & mask5;
            memxor_lop7<N>(matrix + r * stride,
                            caches[0] + (w0 << logwidth),
                            caches[1] + (w1 << logwidth),
                            caches[2] + (w2 << logwidth),
                            caches[3] + (w3 << logwidth),
                            caches[4] + (w4 << logwidth),
                            caches[5] + (w5 << logwidth));
        }
    }
}


static bool find_pivots(uint64_t* RESTRICT pivots, uint64_t* RESTRICT this_strip, uint64_t rows, uint64_t &starting_row, uint64_t *workspace, uint64_t &next_b, uint64_t final_b, int K, int& k) {

    // sorted copy, so that we can skip existing pivots:
    uint64_t spivots[64] = {(uint64_t) -1};

    // find pivots
    uint64_t b = 0;

    while (k < K) {

        int l = 0;
        b = ((uint64_t) -1);

        for (uint64_t s = starting_row; s < rows; s++) {

            if (s == spivots[l]) {
                // don't use an existing pivot:
                l += 1; continue;
            }

            uint64_t this_row = this_strip[s];
            uint64_t a = (this_row & (-this_row)) - 1;
            if (a < b) {
                b = a;
                pivots[k] = s;
                if (b == next_b) {
                    // we've found the best pivot possible:
                    break;
                }
            }
        }

        if (b == ((uint64_t) -1)) {
            // we have exhausted this strip with no pivot found:
            return true;
        }

        uint64_t j = pivots[k];
        uint64_t wsj = workspace[j] ^ (1ull << k);
        uint64_t m = this_strip[j];
        uint64_t ml = m & (-m);

        for (uint64_t s = 0; s < rows; s++) {
            if (s == j) { continue; }
            if (this_strip[s] & ml) {
                this_strip[s] ^= m;
                workspace[s] ^= wsj;
            }
        }

        spivots[k] = pivots[k];
        l = k;
        while (l --> 0) {
            // insertion sort:
            if (spivots[l] > spivots[l+1]) {
                uint64_t x = spivots[l];
                spivots[l] = spivots[l+1];
                spivots[l+1] = x;
            }
        }

        k += 1;
        next_b = (b << 1) + 1;
        if (b == final_b) {
            // we have found a pivot for the last column in this strip:
            return true;
        }
    }

    // we have found K pivots and have not proved that this 64-column strip
    // has been fully exhausted:
    return false;
}

/**
 * Use Kronrod's algorithm to reduce all strips to the right of the current
 * strip. We do this in chunks of between 1 and 32 strips (64 to 2048 columns)
 * and attempt to align chunks with cache lines if the stride is a multiple
 * of the cache line size.
 *
 * The long switch statements are because we generate bespoke code for each
 * value of the chunk width N, which outperforms having a variable-length loop.
 */
static void chunked_kronrod(const uint64_t* RESTRICT pivots, uint64_t* RESTRICT matrix, uint64_t rows, uint64_t strips, uint64_t stride, const uint64_t* workspace, uint64_t* RESTRICT cache, int k) {

    uint64_t re = strips - 1;

    #define KRONROD(N) kronrod<N>(matrix + (strips - re), rows, stride, workspace, cache, pivots, k)

    if ((re > 32) && ((stride & 7) == 0)) {
        // try to optimise for cache lines:
        uint64_t ptr = ((uint64_t) (matrix + (strips - re)));

        // optimise for both 64-byte and 128-byte cache lines:
        uint64_t mask = (stride - 1) & 15; // either 0b0111 or 0b1111
        uint64_t ideal_re = 16 - ((ptr >> 3) & mask);

        switch (ideal_re) {
            case 15: KRONROD(15); re -= 15; break;
            case 14: KRONROD(14); re -= 14; break;
            case 13: KRONROD(13); re -= 13; break;
            case 12: KRONROD(12); re -= 12; break;
            case 11: KRONROD(11); re -= 11; break;
            case 10: KRONROD(10); re -= 10; break;
            case  9: KRONROD( 9); re -=  9; break;
            case  8: KRONROD( 8); re -=  8; break;
            case  7: KRONROD( 7); re -=  7; break;
            case  6: KRONROD( 6); re -=  6; break;
            case  5: KRONROD( 5); re -=  5; break;
            case  4: KRONROD( 4); re -=  4; break;
            case  3: KRONROD( 3); re -=  3; break;
            case  2: KRONROD( 2); re -=  2; break;
            case  1: KRONROD( 1); re -=  1; break;
        }
    }

    while (re >= 32) {
        KRONROD(32);
        re -= 32;
    }

    if (re >= 16) {
        KRONROD(16);
        re -= 16;
    }

    switch (re) {
        // process the last (incomplete) chunk:
        case 15: KRONROD(15); break;
        case 14: KRONROD(14); break;
        case 13: KRONROD(13); break;
        case 12: KRONROD(12); break;
        case 11: KRONROD(11); break;
        case 10: KRONROD(10); break;
        case  9: KRONROD( 9); break;
        case  8: KRONROD( 8); break;
        case  7: KRONROD( 7); break;
        case  6: KRONROD( 6); break;
        case  5: KRONROD( 5); break;
        case  4: KRONROD( 4); break;
        case  3: KRONROD( 3); break;
        case  2: KRONROD( 2); break;
        case  1: KRONROD( 1); break;
    }

    #undef KRONROD
}


/**
 * Find up to K pivot rows in this strip of 64 columns, remove them from all
 * other rows, and permute them into the correct places.
 */
static bool perform_K_steps(uint64_t* RESTRICT matrix, uint64_t* RESTRICT stripspace, uint64_t rows, uint64_t strips, uint64_t stride, uint64_t &starting_row, uint64_t *workspace, uint64_t* RESTRICT cache, uint64_t &next_b, int K, uint64_t final_b) {

    memset(workspace, 0, 8 * rows);

    // array to contain the indices of the k pivot rows:
    uint64_t pivots[64] = {(uint64_t) -1};

    int k = 0;
    bool completed_strip = find_pivots(pivots, stripspace, rows, starting_row, workspace, next_b, final_b, K, k);

    if (k == 0) {
        // no pivots detected:
        return true;
    }

    for (uint64_t r = 0; r < rows; r++) {
        matrix[r * stride] = stripspace[r];
    }

    // for all strips to the right of the current strip, use Kronrod's
    // method to XOR the correct linear combination of the k pivot rows
    // from each row in the matrix:
    chunked_kronrod(pivots, matrix, rows, strips, stride, workspace, cache, k);

    // apply a row permutation so that the k pivot rows are moved to the
    // uppermost k slots, incrementing starting_row in the process:
    for (int i = 0; i < k; i++) {
        if (pivots[i] != starting_row) {
            // swap rows in matrix:
            swap_rows(matrix + starting_row * stride, matrix + pivots[i] * stride, strips);
            // swap rows in stripspace:
            uint64_t x = stripspace[pivots[i]];
            stripspace[pivots[i]] = stripspace[starting_row];
            stripspace[starting_row] = x;
            for (int j = 0; j < k; j++) {
                if (pivots[j] == starting_row) { pivots[j] = pivots[i]; }
            }
            pivots[i] = starting_row;
        }
        starting_row += 1;
    }

    // determine whether we have exhausted all of the columns in the strip:
    return completed_strip;
}


static void inplace_rref_strided_K(uint64_t* RESTRICT matrix, uint64_t* RESTRICT stripspace, uint64_t rows, uint64_t cols, uint64_t stride, uint64_t *workspace, uint64_t *cache, int K) {

    uint64_t strips = (cols + 63) >> 6;

    uint64_t current_row = 0;

    for (uint64_t current_strip = 0; current_strip < strips; current_strip++) {
        uint64_t remcols = cols - (current_strip << 6);
        if (remcols > 64) { remcols = 64; }
        uint64_t final_b = (1ull << (remcols - 1)) - 1;
        uint64_t next_b = 0;

        uint64_t *offset_matrix = matrix + current_strip;

        // We make a cached copy of the current strip. This has contiguous
        // memory layout (unlike the source strip in the matrix), and the
        // performance gain from having contiguity massively exceeds the
        // cost of copying between the matrix and this cached copy.
        for (uint64_t r = 0; r < rows; r++) {
            stripspace[r] = offset_matrix[r * stride];
        }

        while (current_row < rows) {
            if (perform_K_steps(offset_matrix, stripspace, rows, strips - current_strip, stride, current_row, workspace, cache, next_b, K, final_b)) {
                break;
            }
        }

        if (current_row >= rows) { break; }
    }
}


static void inplace_rref_strided_heap(uint64_t *matrix, uint64_t rows, uint64_t cols, uint64_t stride, int K) {

    // Array for storing, for each row, the appropriate linear combination of
    // the k <= K <= 32 pivot rows that needs to be subtracted:
    uint64_t* workspace = ((uint64_t*) malloc(rows * 8));

    // Array for caching the current strip (64 columns) of the matrix:
    uint64_t* stripspace = ((uint64_t*) malloc(rows * 8));

    int subkays[6];
    split_k(K, subkays);

    // Array for storing 256-byte chunks of linear combinations of pivot rows:
    void* cache_raw = malloc(256 * (1 + (1 << subkays[0]) + (1 << subkays[1]) + (1 << subkays[2]) + (1 << subkays[3]) + (1 << subkays[4]) + (1 << subkays[5])));

    // Align to cache lines:
    uint64_t cache_ptr = ((uint64_t) cache_raw);
    cache_ptr += (128 - (cache_ptr & 127));
    uint64_t* cache = ((uint64_t*) cache_ptr);

    // Convert to row reduced echelon form:
    inplace_rref_strided_K(matrix, stripspace, rows, cols, stride, workspace, cache, K);

    // Free the allocated memory buffers:
    free(workspace);
    free(stripspace);
    free(cache_raw);
}

static void inplace_rref_small(uint64_t *matrix, uint64_t rows, uint64_t cols) {

    uint64_t final_b = (1ull << (cols - 1)) - 1;
    uint64_t next_b = 0;

    for (uint64_t r = 0; r < rows; r++) {

        uint64_t b = (matrix[r] & (-matrix[r])) - 1;

        for (uint64_t s = r+1; s < rows; s++) {
            uint64_t this_row = matrix[s];
            uint64_t a = (this_row & (-this_row)) - 1;

            if (a < b) {
                b = a;
                matrix[s] = matrix[r];
                matrix[r] = this_row;
            }

            if (b == next_b) { break; }
        }

        if (b == ((uint64_t) -1)) { break; }

        uint64_t m = matrix[r];
        uint64_t ml = m & (-m);

        for (uint64_t s = 0; s < rows; s++) {
            if (s == r) { continue; }
            if (matrix[s] & ml) { matrix[s] ^= m; }
        }

        next_b = (b << 1) + 1;
        if (b == final_b) { break; }
    }
}

} // anonymous namespace

namespace f2reduce {

void inplace_rref_strided(uint64_t *matrix, uint64_t rows, uint64_t cols, uint64_t stride) {

    if (rows <= 1 || cols == 0) {
        // If the matrix has 0 or 1 rows or 0 columns, it must already be in RREF:
        return;
    }

    if ((rows <= 64) && (cols <= 64)) {
        uint64_t matrix2[64];
        for (uint64_t i = 0; i < rows; i++) { matrix2[i] = matrix[i * stride]; }
        inplace_rref_small(matrix2, rows, cols);
        for (uint64_t i = 0; i < rows; i++) { matrix[i * stride] = matrix2[i]; }
    } else {
        // Select value of k to minimise the objective function:
        // ceil(64/k) * (rows + 2^(k/2))
        int k = (rows <= 5120) ? 32 : 64;
        inplace_rref_strided_heap(matrix, rows, cols, stride, k);
    }
}

uint64_t get_recommended_stride(uint64_t cols) {

    uint64_t stride = (cols + 63) >> 6;
    if (stride > 32) {
        // pad to a multiple of a 64/128-byte cache line:
        stride += (-stride) & 15;
    }
    if ((stride & 63) == 0) {
        // ensure not divisible by 64 to avoid critical stride issues:
        stride += 16;
    }
    return stride;

}

} // namespace f2reduce
