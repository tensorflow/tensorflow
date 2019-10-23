#ifndef _xc_stdlib_h_
#define _xc_stdlib_h_

#include_next <stdlib.h>
#include <safe/stdlib.h>

#if defined(__XC__) && !defined(UNSAFE_LIBC)
#define atoi(nptr) _safe_atoi(nptr)
#define atol(nptr) _safe_atol(nptr)
#define atoll(nptr) _safe_atoll(nptr)
#define calloc(nmemb, size) _safe_calloc(nmemb, size)
#define free(s) free(s)
#define getenv(string) _safe_getenv(string)
#define malloc(size) _safe_malloc(size)
#define realloc(r, size) _safe_realloc(r, size)
#define strtod(n, endptr) _safe_strtod(n, endptr)
#define strtof(n, endptr) _safe_strtof(n, endptr)
#define strtol(n, endptr, base) _safe_strtol(n, endptr, base)
#define strtoul(n, endptr, base) _safe_strtoul(n, endptr, base)
#define strtoll(n, endptr, base) _safe_strtoll(n, endptr, base)
#define strtoull(n, endptr, base) _safe_strtoull(n, endptr, base)
#define system(string) _safe_system(string)

#endif

#endif /* _xc_stdlib_h_ */
