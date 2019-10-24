#ifndef _xc_locale_h_
#define _xc_locale_h_

#include_next <locale.h>
#include <safe/locale.h>

#if defined(__XC__) && !defined(UNSAFE_LIBC)
#define setlocale(category, locale) _safe_setlocale(category, locale)
#define localeconv() _safe_localeconv(nptr, endptr, base)
#endif

#endif /* _xc_locale_h_ */
