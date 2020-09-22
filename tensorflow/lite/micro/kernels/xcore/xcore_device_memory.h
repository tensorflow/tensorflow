// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_DEVICE_MEMORY_H_
#define XCORE_DEVICE_MEMORY_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef XCORE
#ifdef _TIME_H_
#define _clock_defined
#endif
#include <xcore/thread.h>

#define STRINGIFY(NAME) #NAME
#define GET_STACKWORDS(DEST, NAME) \
  asm("ldc %[__dest], " STRINGIFY(NAME) ".nstackwords" : [ __dest ] "=r"(DEST))
#define GET_STACKSIZE(DEST, NAME)                        \
  {                                                      \
    size_t _stack_words;                                 \
    asm("ldc %[__dest], " STRINGIFY(NAME) ".nstackwords" \
        : [ __dest ] "=r"(_stack_words));                \
    DEST = (_stack_words + 2) * 4;                       \
  }
#define IS_RAM(a) (((uintptr_t)a >= 0x80000) && ((uintptr_t)a <= 0x100000))
#define IS_NOT_RAM(a) ((uintptr_t)a > 0x100000)
#define IS_EXTMEM(a) \
  (((uintptr_t)a >= 0x10000000) && (((uintptr_t)a <= 0x20000000)))
#define IS_SWMEM(a) \
  (((uintptr_t)a >= 0x40000000) && (((uintptr_t)a <= 0x80000000)))

#ifdef USE_SWMEM
#ifndef USE_QSPI_SWMEM_DEV
void swmem_setup();
#else
#include <xcore/chanend.h>
void swmem_setup(chanend_t ctrl_swmem_c);
#endif  // USE_QSPI_SWMEM_DEV
#endif  // USE_SWMEM

void swmem_handler(void *ignored);
void swmem_teardown();

#else  // not XCORE

#define GET_STACKSIZE(DEST, NAME) DEST = 0
#define GET_STACKWORDS(DEST, NAME) DEST = 0
#define IS_RAM(a) (1)
#define IS_NOT_RAM(a) (0)

#endif  // XCORE

void memload(void *dest, void *src, size_t size);

#ifdef __cplusplus
}
#endif

#endif  // XCORE_DEVICE_MEMORY_H_
