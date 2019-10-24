/* timeb.h -- An implementation of the standard Unix <sys/timeb.h> file.
   Written by Ian Lance Taylor <ian@cygnus.com>
   Public domain; no rights reserved.

   <sys/timeb.h> declares the structure used by the ftime function, as
   well as the ftime function itself.  Newlib does not provide an
   implementation of ftime.  */

#ifndef _SYS_TIMEB_H

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

#define _SYS_TIMEB_H

#include <_ansi.h>
#include <machine/types.h>

#ifndef __time_t_defined
typedef _TIME_T_ time_t;
#define __time_t_defined
#endif

struct timeb
{
  time_t time;
  unsigned short millitm;
  short timezone;
  short dstflag;
};

extern int ftime _PARAMS ((struct timeb *));

#if defined(__cplusplus) || defined(__XC__)
}
#endif

#endif /* ! defined (_SYS_TIMEB_H) */
