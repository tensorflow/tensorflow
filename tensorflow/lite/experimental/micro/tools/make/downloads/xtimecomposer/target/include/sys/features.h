/*
 *  Written by Joel Sherrill <joel@OARcorp.com>.
 *
 *  COPYRIGHT (c) 1989-2000.
 *
 *  On-Line Applications Research Corporation (OAR).
 *
 *  Permission to use, copy, modify, and distribute this software for any
 *  purpose without fee is hereby granted, provided that this entire notice
 *  is included in all copies of any software which is or includes a copy
 *  or modification of this software.
 *
 *  THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 *  WARRANTY.  IN PARTICULAR,  THE AUTHOR MAKES NO REPRESENTATION
 *  OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY OF THIS
 *  SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 *
 *  $Id: features.h,v 1.13 2007/12/19 18:51:19 jjohnstn Exp $
 */

#ifndef _SYS_FEATURES_H
#define _SYS_FEATURES_H

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

/* RTEMS adheres to POSIX -- 1003.1b with some features from annexes.  */

#ifdef __rtems__
#define _POSIX_JOB_CONTROL     		1
#define _POSIX_SAVED_IDS       		1
#define _POSIX_VERSION			199309L
#define _POSIX_ASYNCHRONOUS_IO		1
#define _POSIX_FSYNC			1
#define _POSIX_MAPPED_FILES		1
#define _POSIX_MEMLOCK			1
#define _POSIX_MEMLOCK_RANGE		1
#define _POSIX_MEMORY_PROTECTION	1
#define _POSIX_MESSAGE_PASSING		1
#define _POSIX_PRIORITIZED_IO		1
#define _POSIX_PRIORITY_SCHEDULING	1
#define _POSIX_REALTIME_SIGNALS		1
#define _POSIX_SEMAPHORES		1
#define _POSIX_SHARED_MEMORY_OBJECTS	1
#define _POSIX_SYNCHRONIZED_IO		1
#define _POSIX_TIMERS			1
#define _POSIX_BARRIERS                 200112L
#define _POSIX_READER_WRITER_LOCKS      200112L
#define _POSIX_SPIN_LOCKS               200112L


/* In P1003.1b but defined by drafts at least as early as P1003.1c/D10  */
#define _POSIX_THREADS				1
#define _POSIX_THREAD_ATTR_STACKADDR		1
#define _POSIX_THREAD_ATTR_STACKSIZE		1
#define _POSIX_THREAD_PRIORITY_SCHEDULING	1
#define _POSIX_THREAD_PRIO_INHERIT		1
#define _POSIX_THREAD_PRIO_PROTECT		1
#define _POSIX_THREAD_PROCESS_SHARED		1
#define _POSIX_THREAD_SAFE_FUNCTIONS		1

/* P1003.4b/D8 defines the constants below this comment. */
#define _POSIX_SPAWN				1
#define _POSIX_TIMEOUTS				1
#define _POSIX_CPUTIME				1
#define _POSIX_THREAD_CPUTIME			1
#define _POSIX_SPORADIC_SERVER			1
#define _POSIX_THREAD_SPORADIC_SERVER		1
#define _POSIX_DEVICE_CONTROL			1
#define _POSIX_DEVCTL_DIRECTION			1
#define _POSIX_INTERRUPT_CONTROL		1
#define _POSIX_ADVISORY_INFO			1

#endif

#ifdef __svr4__
# define _POSIX_JOB_CONTROL     1
# define _POSIX_SAVED_IDS       1
# define _POSIX_VERSION 199009L
#endif

#ifdef __CYGWIN__

#if !defined(__STRICT_ANSI__) || defined(__cplusplus) || __STDC_VERSION__ >= 199901L
#define _POSIX_VERSION				200112L
#define _POSIX2_VERSION				200112L
#define _XOPEN_VERSION				   600

#define _POSIX_ADVISORY_INFO			200112L
/* #define _POSIX_ASYNCHRONOUS_IO		    -1 */
/* #define _POSIX_BARRIERS			    -1 */
#define _POSIX_CHOWN_RESTRICTED			     1
/* #define _POSIX_CLOCK_SELECTION		    -1 */
/* #define _POSIX_CPUTIME			    -1 */
#define _POSIX_FSYNC				200112L
#define _POSIX_IPV6				200112L
#define _POSIX_JOB_CONTROL			     1
#define _POSIX_MAPPED_FILES			200112L
/* #define _POSIX_MEMLOCK			    -1 */
#define _POSIX_MEMLOCK_RANGE			200112L
#define _POSIX_MEMORY_PROTECTION		200112L
#define _POSIX_MESSAGE_PASSING			200112L
/* #define _POSIX_MONOTONIC_CLOCK		    -1 */
#define _POSIX_NO_TRUNC				     1
/* #define _POSIX_PRIORITIZED_IO		    -1 */
#define _POSIX_PRIORITY_SCHEDULING		200112L
#define _POSIX_RAW_SOCKETS			200112L
#define _POSIX_READER_WRITER_LOCKS		200112L
#define _POSIX_REALTIME_SIGNALS			200112L
#define _POSIX_REGEXP				     1
#define _POSIX_SAVED_IDS			     1
#define _POSIX_SEMAPHORES			200112L
#define _POSIX_SHARED_MEMORY_OBJECTS		200112L 
#define _POSIX_SHELL				     1
/* #define _POSIX_SPAWN				    -1 */
/* #define _POSIX_SPIN_LOCKS			    -1 */
/* #define _POSIX_SPORADIC_SERVER		    -1 */
#define _POSIX_SYNCHRONIZED_IO			200112L
/* #define _POSIX_THREAD_ATTR_STACKADDR		    -1 */
#define _POSIX_THREAD_ATTR_STACKSIZE		200112L
/* #define _POSIX_THREAD_CPUTIME		    -1 */
/* #define _POSIX_THREAD_PRIO_INHERIT		    -1 */
/* #define _POSIX_THREAD_PRIO_PROTECT		    -1 */
#define _POSIX_THREAD_PRIORITY_SCHEDULING	200112L
#define _POSIX_THREAD_PROCESS_SHARED		200112L
#define _POSIX_THREAD_SAFE_FUNCTIONS		200112L
/* #define _POSIX_THREAD_SPORADIC_SERVER	    -1 */
#define _POSIX_THREADS				200112L
/* #define _POSIX_TIMEOUTS			    -1 */
#define _POSIX_TIMERS				     1
/* #define _POSIX_TRACE				    -1 */
/* #define _POSIX_TRACE_EVENT_FILTER		    -1 */
/* #define _POSIX_TRACE_INHERIT			    -1 */
/* #define _POSIX_TRACE_LOG			    -1 */
/* #define _POSIX_TYPED_MEMORY_OBJECTS		    -1 */
#define _POSIX_VDISABLE				   '\0'
#define _POSIX2_C_BIND				200112L
#define _POSIX2_C_DEV				200112L
#define _POSIX2_CHAR_TERM			200112L
/* #define _POSIX2_FORT_DEV			    -1 */
/* #define _POSIX2_FORT_RUN			    -1 */
/* #define _POSIX2_LOCALEDEF			    -1 */
/* #define _POSIX2_PBS				    -1 */
/* #define _POSIX2_PBS_ACCOUNTING		    -1 */
/* #define _POSIX2_PBS_CHECKPOINT		    -1 */
/* #define _POSIX2_PBS_LOCATE			    -1 */
/* #define _POSIX2_PBS_MESSAGE			    -1 */
/* #define _POSIX2_PBS_TRACK			    -1 */
#define _POSIX2_SW_DEV				200112L
#define _POSIX2_UPE				200112L
/* #define _POSIX_V6_ILP32_OFF32		    -1 */
#define _XBS5_ILP32_OFF32			_POSIX_V6_ILP32_OFF32
#define _POSIX_V6_ILP32_OFFBIG			     1
#define _XBS5_ILP32_OFFBIG			_POSIX_V6_ILP32_OFFBIG
/* #define _POSIX_V6_LP64_OFF64			    -1 */
#define _XBS5_LP64_OFF64			_POSIX_V6_LP64_OFF64
/* #define _POSIX_V6_LPBIG_OFFBIG		    -1 */
#define _XBS5_LPBIG_OFFBIG			_POSIX_V6_LPBIG_OFFBIG
#define _XOPEN_CRYPT				     1
#define _XOPEN_ENH_I18N				     1
/* #define _XOPEN_LEGACY			    -1 */
/* #define _XOPEN_REALTIME			    -1 */
/* #define _XOPEN_REALTIME_THREADS		    -1 */
#define _XOPEN_SHM				     1
/* #define _XOPEN_STREAMS			    -1 */
/* #define _XOPEN_UNIX				    -1 */

#endif /* !__STRICT_ANSI__ || __cplusplus || __STDC_VERSION__ >= 199901L */
#endif /* __CYGWIN__ */

#ifdef __SPU__
/* Not much for now! */
#define _POSIX_TIMERS				     1
#endif

#if defined(__cplusplus) || defined(__XC__)
}
#endif
#endif /* _SYS_FEATURES_H */
