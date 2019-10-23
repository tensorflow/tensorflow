// Copyright (c) 2014-2016, XMOS Ltd, All rights reserved

#ifndef __hwlock_h_
#define __hwlock_h_

#include <xs1.h>

#define QUOTEAUX(x) #x
#define QUOTE(x) QUOTEAUX(x)

/** This type represents a hardware lock. */
typedef unsigned hwlock_t;

enum {
  HWLOCK_NOT_ALLOCATED = 0
};

/** Allocate a hardware lock.
 *
 * This function will allocate a new hardware lock from the pool of hardware
 * locks available on the xCORE. The hardware has a limited number of hardware
 * locks (for example, current L and S series devices have 4 locks per tile).
 *
 * \returns  the allocated lock if allocation is successful or the value
 *           ``HWLOCK_NOT_ALLOCATED`` if not.
 */
inline hwlock_t hwlock_alloc(void)
{
  hwlock_t lock;
  asm volatile ("getr %0, " QUOTE(XS1_RES_TYPE_LOCK)
                    : "=r" (lock));
  return lock;
}

/** Free a hardware lock.
 *
 *  This function frees a given hardware lock and returns it to the
 *  hardware pool to be reallocated elsewhere.
 *
 *  \param lock  the hardware lock to be freed. If this is an invalid lock id or
 *               not an currently allocated lock then the function will trap.
 */
inline void hwlock_free(hwlock_t lock)
{
  asm volatile ("freer res[%0]"
                        : /* no output */
                        : "r" (lock));
}

/** Acquire a hardware lock.
 *
 *  This function acquires a lock for the current logical core. If
 *  another core holds the lock the function will pause until the
 *  lock is released.
 *
 *  \param lock  the hardware lock to acquire
 */
inline void hwlock_acquire(hwlock_t lock)
{
  asm volatile ("in %0, res[%0]"
                        : /* no output */
                        : "r" (lock)
                        : "memory");
}

/** Release a hardware lock.
 *
 * This function releases a lock from the current logical core. The lock should
 * have been previously claimed by hwlock_acquire().
 *
 * \param lock  the hardware lock to release
 */
inline void hwlock_release(hwlock_t lock)
{
  asm volatile ("out res[%0], %0"
                        : /* no output */
                        : "r" (lock)
                        : "memory");
}

#endif // __hwlock_h_
