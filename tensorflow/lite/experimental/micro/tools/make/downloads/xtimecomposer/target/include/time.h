/*
 * time.h
 * 
 * Struct and function declarations for dealing with time.
 */

#ifndef _TIME_H_
#define _TIME_H_

#include "_ansi.h"

#include <sys/reent.h>

#ifndef NULL
#define	NULL	0
#endif

/* Get _CLOCKS_PER_SEC_ */
#include <machine/time.h>

#ifndef _CLOCKS_PER_SEC_
#define _CLOCKS_PER_SEC_ 1000
#endif

#define CLOCKS_PER_SEC _CLOCKS_PER_SEC_
#define CLK_TCK CLOCKS_PER_SEC
#define __need_size_t
#include <stddef.h>

#include <sys/types.h>

_BEGIN_STD_C

struct tm
{
  int	tm_sec;
  int	tm_min;
  int	tm_hour;
  int	tm_mday;
  int	tm_mon;
  int	tm_year;
  int	tm_wday;
  int	tm_yday;
  int	tm_isdst;
};

clock_t	   _EXFUN(clock,    (void));
double	   _EXFUN(difftime, (time_t _time2, time_t _time1));
time_t	   _EXFUN(mktime,   (struct tm *_timeptr));
time_t	   _EXFUN(time,     (time_t *_timer));
char	  *_EXFUN(asctime,  (const struct tm *_tblock));
char	  *_EXFUN(ctime,    (const time_t *_time));
struct tm *_EXFUN(gmtime,   (const time_t *_timer));
struct tm *_EXFUN(localtime,(const time_t *_timer));
size_t	   _EXFUN(strftime, (char *_s, size_t _maxsize, const char *_fmt, const struct tm *_t));

char	  *_EXFUN(asctime_r,	(const struct tm *, char *));
char	  *_EXFUN(ctime_r,	(const time_t *, char *));
struct tm *_EXFUN(gmtime_r,	(const time_t *, struct tm *));
struct tm *_EXFUN(localtime_r,	(const time_t *, struct tm *));

_END_STD_C

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

#ifndef __STRICT_ANSI__
char      *_EXFUN(strptime,     (const char *, const char *, struct tm *));
_VOID      _EXFUN(tzset,	(_VOID));
_VOID      _EXFUN(_tzset_r,	(struct _reent *));

typedef struct __tzrule_struct
{
  char ch;
  int m;
  int n;
  int d;
  int s;
  time_t change;
  long offset; /* Match type of _timezone. */
} __tzrule_type;

typedef struct __tzinfo_struct
{
  int __tznorth;
  int __tzyear;
  __tzrule_type __tzrule[2];
} __tzinfo_type;

__tzinfo_type *_EXFUN (__gettzinfo, (_VOID));

/* getdate functions */

#ifdef HAVE_GETDATE
#define getdate_err (*__getdate_err())
int *_EXFUN(__getdate_err,(_VOID));

struct tm *	_EXFUN(getdate, (const char *));
/* getdate_err is set to one of the following values to indicate the error.
     1  the DATEMSK environment variable is null or undefined,
     2  the template file cannot be opened for reading,
     3  failed to get file status information,
     4  the template file is not a regular file,
     5  an error is encountered while reading the template file,
     6  memory allication failed (not enough memory available),
     7  there is no line in the template that matches the input,
     8  invalid input specification  */

/* getdate_r returns the error code as above */
int		_EXFUN(getdate_r, (const char *, struct tm *));
#endif /* HAVE_GETDATE */

/* defines for the opengroup specifications Derived from Issue 1 of the SVID.  */
extern __IMPORT long _timezone;
extern __IMPORT int _daylight;
extern __IMPORT char *_tzname[2];

/* POSIX defines the external tzname being defined in time.h */
#ifndef tzname
#define tzname _tzname
#endif
#endif /* !__STRICT_ANSI__ */

#if defined(__cplusplus) || defined(__XC__)
}
#endif

#include <sys/features.h>

#ifdef __CYGWIN__
#include <cygwin/time.h>
#endif /*__CYGWIN__*/

#if defined(_POSIX_TIMERS)

#include <signal.h>

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

/* Clocks, P1003.1b-1993, p. 263 */

int _EXFUN(clock_settime, (clockid_t clock_id, const struct timespec *tp));
int _EXFUN(clock_gettime, (clockid_t clock_id, struct timespec *tp));
int _EXFUN(clock_getres,  (clockid_t clock_id, struct timespec *res));

/* Create a Per-Process Timer, P1003.1b-1993, p. 264 */

int _EXFUN(timer_create,
  (clockid_t clock_id, struct sigevent *evp, timer_t *timerid));

/* Delete a Per_process Timer, P1003.1b-1993, p. 266 */

int _EXFUN(timer_delete, (timer_t timerid));

/* Per-Process Timers, P1003.1b-1993, p. 267 */

int _EXFUN(timer_settime,
  (timer_t timerid, int flags, const struct itimerspec *value,
   struct itimerspec *ovalue));
int _EXFUN(timer_gettime, (timer_t timerid, struct itimerspec *value));
int _EXFUN(timer_getoverrun, (timer_t timerid));

/* High Resolution Sleep, P1003.1b-1993, p. 269 */

int _EXFUN(nanosleep, (const struct timespec  *rqtp, struct timespec *rmtp));

#if defined(__cplusplus) || defined(__XC__)
}
#endif
#endif /* _POSIX_TIMERS */

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

/* CPU-time Clock Attributes, P1003.4b/D8, p. 54 */

/* values for the clock enable attribute */

#define CLOCK_ENABLED  1  /* clock is enabled, i.e. counting execution time */
#define CLOCK_DISABLED 0  /* clock is disabled */

/* values for the pthread cputime_clock_allowed attribute */

#define CLOCK_ALLOWED    1 /* If a thread is created with this value a */
                           /*   CPU-time clock attached to that thread */
                           /*   shall be accessible. */
#define CLOCK_DISALLOWED 0 /* If a thread is created with this value, the */
                           /*   thread shall not have a CPU-time clock */
                           /*   accessible. */

/* Manifest Constants, P1003.1b-1993, p. 262 */

#define CLOCK_REALTIME (clockid_t)1

/* Flag indicating time is "absolute" with respect to the clock
   associated with a time.  */

#define TIMER_ABSTIME	4

/* Manifest Constants, P1003.4b/D8, p. 55 */

#if defined(_POSIX_CPUTIME)

/* When used in a clock or timer function call, this is interpreted as
   the identifier of the CPU_time clock associated with the PROCESS
   making the function call.  */

#define CLOCK_PROCESS_CPUTIME (clockid_t)2

#endif

#if defined(_POSIX_THREAD_CPUTIME)

/*  When used in a clock or timer function call, this is interpreted as
    the identifier of the CPU_time clock associated with the THREAD
    making the function call.  */

#define CLOCK_THREAD_CPUTIME (clockid_t)3

#endif

#if defined(_POSIX_CPUTIME)

/* Accessing a Process CPU-time CLock, P1003.4b/D8, p. 55 */

int _EXFUN(clock_getcpuclockid, (pid_t pid, clockid_t *clock_id));

#endif /* _POSIX_CPUTIME */

#if defined(_POSIX_CPUTIME) || defined(_POSIX_THREAD_CPUTIME)

/* CPU-time Clock Attribute Access, P1003.4b/D8, p. 56 */

int _EXFUN(clock_setenable_attr, (clockid_t clock_id, int attr));
int _EXFUN(clock_getenable_attr, (clockid_t clock_id, int *attr));

#endif /* _POSIX_CPUTIME or _POSIX_THREAD_CPUTIME */

#if defined(__cplusplus) || defined(__XC__)
}
#endif

#endif /* _TIME_H_ */

