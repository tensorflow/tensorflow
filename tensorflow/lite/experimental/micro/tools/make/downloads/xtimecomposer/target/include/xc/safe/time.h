#ifndef _xc_safe_time_h_
#define _xc_safe_time_h_

#include <time.h>

#ifdef __XC__
time_t _safe_mktime(struct tm timep[1]);
time_t _safe_time(time_t (&?timep)[1]);
char * alias _safe_asctime(const struct tm tblock[1]);
char * alias _safe_ctime(const time_t time[1]);
struct tm * alias _safe_gmtime(const time_t timep[1]);
struct tm * alias _safe_localtime(const time_t timep[1]);
size_t _safe_strftime(char s[maxsize], size_t maxsize, const char fmt[], const struct tm t[1]);
#endif

#endif /* _xc_safe_time_h_ */
