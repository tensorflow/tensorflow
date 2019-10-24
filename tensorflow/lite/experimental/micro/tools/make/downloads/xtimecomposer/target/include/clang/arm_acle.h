/*===---- arm_acle.h - ARM Non-Neon intrinsics -----------------------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __ARM_ACLE_H
#define __ARM_ACLE_H

#ifndef __ARM_ACLE
#error "ACLE intrinsics support not enabled."
#endif

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* 8 SYNCHRONIZATION, BARRIER AND HINT INTRINSICS */
/* 8.3 Memory barriers */
#if !defined(_MSC_VER)
#define __dmb(i) __builtin_arm_dmb(i)
#define __dsb(i) __builtin_arm_dsb(i)
#define __isb(i) __builtin_arm_isb(i)
#endif

/* 8.4 Hints */

#if !defined(_MSC_VER)
static __inline__ void __attribute__((always_inline, nodebug)) __wfi(void) {
  __builtin_arm_wfi();
}

static __inline__ void __attribute__((always_inline, nodebug)) __wfe(void) {
  __builtin_arm_wfe();
}

static __inline__ void __attribute__((always_inline, nodebug)) __sev(void) {
  __builtin_arm_sev();
}

static __inline__ void __attribute__((always_inline, nodebug)) __sevl(void) {
  __builtin_arm_sevl();
}

static __inline__ void __attribute__((always_inline, nodebug)) __yield(void) {
  __builtin_arm_yield();
}
#endif

#if __ARM_32BIT_STATE
#define __dbg(t) __builtin_arm_dbg(t)
#endif

/* 8.5 Swap */
static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __swp(uint32_t x, volatile uint32_t *p) {
  uint32_t v;
  do v = __builtin_arm_ldrex(p); while (__builtin_arm_strex(x, p));
  return v;
}

/* 8.6 Memory prefetch intrinsics */
/* 8.6.1 Data prefetch */
#define __pld(addr) __pldx(0, 0, 0, addr)

#if __ARM_32BIT_STATE
#define __pldx(access_kind, cache_level, retention_policy, addr) \
  __builtin_arm_prefetch(addr, access_kind, 1)
#else
#define __pldx(access_kind, cache_level, retention_policy, addr) \
  __builtin_arm_prefetch(addr, access_kind, cache_level, retention_policy, 1)
#endif

/* 8.6.2 Instruction prefetch */
#define __pli(addr) __plix(0, 0, addr)

#if __ARM_32BIT_STATE
#define __plix(cache_level, retention_policy, addr) \
  __builtin_arm_prefetch(addr, 0, 0)
#else
#define __plix(cache_level, retention_policy, addr) \
  __builtin_arm_prefetch(addr, 0, cache_level, retention_policy, 0)
#endif

/* 8.7 NOP */
static __inline__ void __attribute__((always_inline, nodebug)) __nop(void) {
  __builtin_arm_nop();
}

/* 9 DATA-PROCESSING INTRINSICS */
/* 9.2 Miscellaneous data-processing intrinsics */
/* ROR */
static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __ror(uint32_t x, uint32_t y) {
  y %= 32;
  if (y == 0)  return x;
  return (x >> y) | (x << (32 - y));
}

static __inline__ uint64_t __attribute__((always_inline, nodebug))
  __rorll(uint64_t x, uint32_t y) {
  y %= 64;
  if (y == 0)  return x;
  return (x >> y) | (x << (64 - y));
}

static __inline__ unsigned long __attribute__((always_inline, nodebug))
  __rorl(unsigned long x, uint32_t y) {
#if __SIZEOF_LONG__ == 4
  return __ror(x, y);
#else
  return __rorll(x, y);
#endif
}


/* CLZ */
static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __clz(uint32_t t) {
  return __builtin_clz(t);
}

static __inline__ unsigned long __attribute__((always_inline, nodebug))
  __clzl(unsigned long t) {
  return __builtin_clzl(t);
}

static __inline__ uint64_t __attribute__((always_inline, nodebug))
  __clzll(uint64_t t) {
  return __builtin_clzll(t);
}

/* REV */
static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __rev(uint32_t t) {
  return __builtin_bswap32(t);
}

static __inline__ unsigned long __attribute__((always_inline, nodebug))
  __revl(unsigned long t) {
#if __SIZEOF_LONG__ == 4
  return __builtin_bswap32(t);
#else
  return __builtin_bswap64(t);
#endif
}

static __inline__ uint64_t __attribute__((always_inline, nodebug))
  __revll(uint64_t t) {
  return __builtin_bswap64(t);
}

/* REV16 */
static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __rev16(uint32_t t) {
  return __ror(__rev(t), 16);
}

static __inline__ unsigned long __attribute__((always_inline, nodebug))
  __rev16l(unsigned long t) {
    return __rorl(__revl(t), sizeof(long) / 2);
}

static __inline__ uint64_t __attribute__((always_inline, nodebug))
  __rev16ll(uint64_t t) {
  return __rorll(__revll(t), 32);
}

/* REVSH */
static __inline__ int16_t __attribute__((always_inline, nodebug))
  __revsh(int16_t t) {
  return __builtin_bswap16(t);
}

/* RBIT */
static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __rbit(uint32_t t) {
  return __builtin_arm_rbit(t);
}

static __inline__ uint64_t __attribute__((always_inline, nodebug))
  __rbitll(uint64_t t) {
#if __ARM_32BIT_STATE
  return (((uint64_t) __builtin_arm_rbit(t)) << 32) |
    __builtin_arm_rbit(t >> 32);
#else
  return __builtin_arm_rbit64(t);
#endif
}

static __inline__ unsigned long __attribute__((always_inline, nodebug))
  __rbitl(unsigned long t) {
#if __SIZEOF_LONG__ == 4
  return __rbit(t);
#else
  return __rbitll(t);
#endif
}

/*
 * 9.4 Saturating intrinsics
 *
 * FIXME: Change guard to their corrosponding __ARM_FEATURE flag when Q flag
 * intrinsics are implemented and the flag is enabled.
 */
/* 9.4.1 Width-specified saturation intrinsics */
#if __ARM_32BIT_STATE
#define __ssat(x, y) __builtin_arm_ssat(x, y)
#define __usat(x, y) __builtin_arm_usat(x, y)
#endif

/* 9.4.2 Saturating addition and subtraction intrinsics */
#if __ARM_32BIT_STATE
static __inline__ int32_t __attribute__((always_inline, nodebug))
  __qadd(int32_t t, int32_t v) {
  return __builtin_arm_qadd(t, v);
}

static __inline__ int32_t __attribute__((always_inline, nodebug))
  __qsub(int32_t t, int32_t v) {
  return __builtin_arm_qsub(t, v);
}

static __inline__ int32_t __attribute__((always_inline, nodebug))
__qdbl(int32_t t) {
  return __builtin_arm_qadd(t, t);
}
#endif

/* 9.7 CRC32 intrinsics */
#if __ARM_FEATURE_CRC32
static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32b(uint32_t a, uint8_t b) {
  return __builtin_arm_crc32b(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32h(uint32_t a, uint16_t b) {
  return __builtin_arm_crc32h(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32w(uint32_t a, uint32_t b) {
  return __builtin_arm_crc32w(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32d(uint32_t a, uint64_t b) {
  return __builtin_arm_crc32d(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32cb(uint32_t a, uint8_t b) {
  return __builtin_arm_crc32cb(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32ch(uint32_t a, uint16_t b) {
  return __builtin_arm_crc32ch(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32cw(uint32_t a, uint32_t b) {
  return __builtin_arm_crc32cw(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32cd(uint32_t a, uint64_t b) {
  return __builtin_arm_crc32cd(a, b);
}
#endif

#if defined(__cplusplus)
}
#endif

#endif /* __ARM_ACLE_H */
