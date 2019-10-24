#ifndef _xc_safe_locale_h_
#define _xc_safe_locale_h_

#include <locale.h>

#ifdef __XC__
char * alias _safe_setlocale(int category, const char (&?locale)[]);
struct lconv * alias _safe_localeconv(void);
#endif

#endif /* _xc_safe_locale_h_ */
