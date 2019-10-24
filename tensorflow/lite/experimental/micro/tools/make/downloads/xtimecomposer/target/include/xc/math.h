#ifndef _xc_math_h_
#define _xc_math_h_

#include_next <math.h>
#include <safe/math.h>

#if defined(__XC__) && !defined(UNSAFE_LIBC)
#define frexp(x, exp) _safe_frexp(x, exp)
#define modf(x, iptr) _safe_modf(x, iptr)
#define nan(tagp) _safe_nan(tagp)
#define remquo(x, y, quo) _safe_remquo(x, y, quo

#define frexpf(x, exp) _safe_frexpf(x, exp)
#define modff(x, iptr) _safe_modff(x, iptr)
#define nanf(tagp) _safe_nanf(tagp)
#define remquof(x, y, quo) _safe_remquof(x, y, quo)
#endif

#endif /* _xc_math_h_ */
