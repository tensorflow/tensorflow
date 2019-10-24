#ifndef _xc_time_h_
#define _xc_time_h_

#include_next <time.h>
#include <safe/time.h>

#if defined(__XC__) && !defined(UNSAFE_LIBC)
#define mktime(timep) _safe_mktime(timep)
#define time(timep) _safe_time(timep)
#define asctime(tblock) _safe_asctime(tblock)
#define ctime(timep) _safe_ctime(timep)
#define gmtime(timep) _safe_gmtime(timep)
#define localtime(timep) _safe_localtime(timep)
#define strftime(s, maxsize, fmt, t) _safe_strftime(s, maxsize, fmt, t)
#endif

#endif /* _xc_time_h_ */
