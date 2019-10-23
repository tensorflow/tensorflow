#ifndef _xc_safe_math_h_
#define _xc_safe_math_h_

#include <math.h>

#ifdef __XC__
extern double _safe_frexp(double x, int exp[1]);
extern double _safe_modf(double x, double iptr[1]);
extern double _safe_nan(const char tagp[]);
extern double _safe_remquo(double x, double y, int quo[1]);

extern float _safe_frexpf(float x, int exp[1]);
extern float _safe_modff(float x, float iptr[1]);
extern float _safe_nanf(const char tagp[]);
extern float _safe_remquof(float x, float y, int quo[1]);
#endif

#endif /* _xc_safe_math_h_ */
