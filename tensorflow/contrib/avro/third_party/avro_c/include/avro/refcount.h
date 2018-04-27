/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.  See the License for the specific language governing
 * permissions and limitations under the License.
 */

#ifndef AVRO_REFCOUNT_H
#define AVRO_REFCOUNT_H

#if defined(_WIN32) && defined(__cplusplus)
/* Include the C++ file <intrin.h> outside the scope of extern "C" */
#include <intrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#define CLOSE_EXTERN }
#else
#define CLOSE_EXTERN
#endif

/**
 * Atomically sets the value of a reference count.
 */

static inline void
avro_refcount_set(volatile int *refcount, int value);

/**
 * Increments a reference count, ensuring that its value doesn't
 * overflow.
 */

static inline void
avro_refcount_inc(volatile int *refcount);

/**
 * Decrements a reference count, and returns whether the resulting
 * (decremented) value is 0.
 */

static inline int
avro_refcount_dec(volatile int *refcount);


/*-----------------------------------------------------------------------
 * Non-Atomic Reference Count
 */
#if defined(AVRO_NON_ATOMIC_REFCOUNT)
static inline void
avro_refcount_set(volatile int *refcount, int value)
{
	*refcount = value;
}

static inline void
avro_refcount_inc(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		*refcount += 1;
	}
}

static inline int
avro_refcount_dec(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		*refcount -= 1;
		return (*refcount == 0);
	}
	return 0;
}

/*-----------------------------------------------------------------------
 * Mac OS X
 */

#elif __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 1050

#include <libkern/OSAtomic.h>

static inline void
avro_refcount_set(volatile int *refcount, int value)
{
	*refcount = value;
}

static inline void
avro_refcount_inc(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		OSAtomicIncrement32(refcount);
	}
}

static inline int
avro_refcount_dec(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		return (OSAtomicDecrement32(refcount) == 0);
	}
	return 0;
}


/*-----------------------------------------------------------------------
 * GCC intrinsics
 */

#elif (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40500

static inline void
avro_refcount_set(volatile int *refcount, int value)
{
	*refcount = value;
}

static inline void
avro_refcount_inc(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		__sync_add_and_fetch(refcount, 1);
	}
}

static inline int
avro_refcount_dec(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		return (__sync_sub_and_fetch(refcount, 1) == 0);
	}
	return 0;
}


/*-----------------------------------------------------------------------
 * Raw x86 assembly
 */

#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))

/* determine the size of int */

#include <limits.h>
#include <avro/platform.h>
#if INT_MAX == INT32_MAX
#define REFCOUNT_SS "l"
#elif INT_MAX == INT64_MAX
#define REFCOUNT_SS "q"
#else
#error "Unknown int size"
#endif

static inline void
avro_refcount_set(volatile int *refcount, int value)
{
	*refcount = value;
}

static inline void
avro_refcount_inc(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		__asm__ __volatile__ ("lock ; inc"REFCOUNT_SS" %0"
				      :"=m" (*refcount)
				      :"m" (*refcount));
	}
}

static inline int
avro_refcount_dec(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		char result;
		__asm__ __volatile__ ("lock ; dec"REFCOUNT_SS" %0; setz %1"
				      :"=m" (*refcount), "=q" (result)
				      :"m" (*refcount));
		return result;
	}
	return 0;
}

#undef REFCOUNT_SS


/*-----------------------------------------------------------------------
 * Raw PPC assembly
 */

#elif defined(__GNUC__) && defined(__ppc__)

static inline int
avro_refcount_LL_int(volatile int *ptr)
{
	int val;
	__asm__ __volatile__ ("lwarx %[val],0,%[ptr]"
			      : [val] "=r" (val)
			      : [ptr] "r" (&ptr)
			      : "cc");

	return val;
}

/* Returns non-zero if the store was successful, zero otherwise. */
static inline int
avro_refcount_SC_int(volatile int *ptr, int val)
{
	int ret = 1; /* init to non-zero, will be reset to 0 if SC was successful */
	__asm__ __volatile__ ("stwcx. %[val],0,%[ptr];\n"
			      "beq 1f;\n"
			      "li %[ret], 0;\n"
			      "1: ;\n"
			      : [ret] "=r" (ret)
			      : [ptr] "r" (&ptr), [val] "r" (val), "0" (ret)
			      : "cc", "memory");
	return ret;
}

static inline void
avro_refcount_set(volatile int *refcount, int value)
{
	*refcount = value;
}

static inline void
avro_refcount_inc(volatile int *refcount)
{
	int prev;
	do {
		prev = avro_refcount_LL_int(refcount);
		if (prev == (int) -1) {
			return;
		}
	} while (!avro_refcount_SC_int(refcount, prev + 1));
}

static inline int
avro_refcount_dec(volatile int *refcount)
{
	int prev;
	do {
		prev = avro_refcount_LL_int(refcount);
		if (prev == (int) -1) {
			return 0;
		}
	} while (!avro_refcount_SC_int(refcount, prev - 1));
	return prev == 1;
}


/*-----------------------------------------------------------------------
 * Windows intrinsics
 */
#elif defined(_WIN32)

#ifdef __cplusplus
// Note: <intrin.h> included outside the extern "C" wrappers above
#else
#include <windows.h>
#include <intrin.h>
#endif // __cplusplus

static inline void
avro_refcount_set(volatile int *refcount, int value)
{
	*refcount = value;
}

static inline void
avro_refcount_inc(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		_InterlockedIncrement((volatile long *) refcount);
	}
}

static inline int
avro_refcount_dec(volatile int *refcount)
{
	if (*refcount != (int) -1) {
		return (_InterlockedDecrement((volatile long *) refcount) == 0);
	}
	return 0;
}

/*-----------------------------------------------------------------------
 * Fallback
 */
#else
#error "No atomic implementation!"
#endif

CLOSE_EXTERN
#endif
