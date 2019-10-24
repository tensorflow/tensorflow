#ifndef _xc_safe_inttypes_h_
#define _xc_safe_inttypes_h_

#include <inttypes.h>

#ifdef __XC__
extern intmax_t _safe_strtoimax(const char nptr[], char * unsafe (&?endptr)[1], int base);
extern uintmax_t _safe_strtoumax(const char nptr[], char * unsafe (&?endptr)[1], int base);
#endif

#endif /* _xc_safe_inttypes_h_ */
