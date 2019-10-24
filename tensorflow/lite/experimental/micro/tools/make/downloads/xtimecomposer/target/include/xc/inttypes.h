#ifndef _xc_inttypes_h_
#define _xc_inttypes_h_

#include_next <inttypes.h>
#include <safe/inttypes.h>

#if defined(__XC__) && !defined(UNSAFE_LIBC)
#define strtoimax(nptr, endptr, base) _safe_strtoimax(nptr, endptr, base)
#define strtoumax(nptr, endptr, base) _safe_strtoumax(nptr, endptr, base)
#endif

#endif /* _xc_inttypes_h_ */
