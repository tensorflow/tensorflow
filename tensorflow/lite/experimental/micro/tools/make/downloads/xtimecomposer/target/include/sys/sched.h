/*
 *  Written by Joel Sherrill <joel@OARcorp.com>.
 *
 *  COPYRIGHT (c) 1989-2000.
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
 *  $Id: sched.h,v 1.2 2002/06/20 19:51:24 fitzsim Exp $
 */


#ifndef __POSIX_SYS_SCHEDULING_h
#define __POSIX_SYS_SCHEDULING_h

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

#include <sys/unistd.h>

#include <sys/types.h>
#include <sys/time.h>

/* Scheduling Policies, P1003.1b-1993, p. 250
   NOTE:  SCHED_SPORADIC added by P1003.4b/D8, p. 34.  */

#define SCHED_OTHER    0
#define SCHED_FIFO     1
#define SCHED_RR       2

#if defined(_POSIX_SPORADIC_SERVER)
#define SCHED_SPORADIC 3 
#endif

/* Scheduling Parameters, P1003.1b-1993, p. 249
   NOTE:  Fields whose name begins with "ss_" added by P1003.4b/D8, p. 33.  */

struct sched_param {
  int sched_priority;           /* Process execution scheduling priority */

#if defined(_POSIX_SPORADIC_SERVER)
  int ss_low_priority;          /* Low scheduling priority for sporadic */
                                /*   server */
  struct timespec ss_replenish_period; 
                                /* Replenishment period for sporadic server */
  struct timespec ss_initial_budget;   /* Initial budget for sporadic server */
#endif
};

#if defined(__cplusplus) || defined(__XC__)
}
#endif 

#endif
/* end of include file */

