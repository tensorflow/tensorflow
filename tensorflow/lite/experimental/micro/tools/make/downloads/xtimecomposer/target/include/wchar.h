#ifndef _WCHAR_H_
#define _WCHAR_H_

#include <_ansi.h>

#include <sys/reent.h>

#define __need_size_t
#define __need_wchar_t
#define __need_wint_t
#include <stddef.h>

/* For _mbstate_t definition. */
#include <sys/_types.h>

#ifndef NULL
#define NULL	0
#endif

#ifndef WEOF
# define WEOF ((wint_t)-1)
#endif

#ifndef WCHAR_MIN
#define WCHAR_MIN 0
#endif

#ifndef WCHAR_MAX
#ifdef __WCHAR_MAX__
#define WCHAR_MAX __WCHAR_MAX__
#else
#define WCHAR_MAX 0x7fffffffu
#endif
#endif

#if !defined(__XC__)

_BEGIN_STD_C

#ifndef _MBSTATE_T
#define _MBSTATE_T
typedef _mbstate_t mbstate_t;
#endif /* _MBSTATE_T */

wint_t	_EXFUN(btowc, (int));
int	_EXFUN(wctob, (wint_t));
size_t	_EXFUN(mbrlen, (const char * , size_t, mbstate_t *));
size_t	_EXFUN(mbrtowc, (wchar_t * , const char * , size_t, mbstate_t *));
size_t	_EXFUN(_mbrtowc_r, (struct _reent *, wchar_t * , const char * , 
			size_t, mbstate_t *));
int	_EXFUN(mbsinit, (const mbstate_t *));
size_t  _EXFUN(mbsnrtowcs, (wchar_t * , const char ** , size_t, size_t,
      mbstate_t *));
size_t  _EXFUN(_mbsnrtowcs_r, (struct _reent *, wchar_t * , const char ** ,
      size_t, size_t, mbstate_t *));
size_t	_EXFUN(mbsrtowcs, (wchar_t * , const char ** , size_t, mbstate_t *));
size_t	_EXFUN(wcrtomb, (char * , wchar_t, mbstate_t *));
size_t	_EXFUN(_wcrtomb_r, (struct _reent *, char * , wchar_t, mbstate_t *));
size_t  _EXFUN(wcsnrtombs, (char * , const wchar_t ** , size_t, size_t,
      mbstate_t *));
size_t  _EXFUN(_wcsnrtombs_r, (struct _reent *, char * , const wchar_t ** ,
      size_t, size_t, mbstate_t *));
size_t	_EXFUN(wcsrtombs, (char * , const wchar_t ** , size_t, mbstate_t *));
size_t	_EXFUN(_wcsrtombs_r, (struct _reent *, char * , const wchar_t ** , 
			size_t, mbstate_t *));
wchar_t	*_EXFUN(wcscat, (wchar_t * , const wchar_t *));
wchar_t	*_EXFUN(wcschr, (const wchar_t *, wchar_t));
int	_EXFUN(wcscmp, (const wchar_t *, const wchar_t *));
int	_EXFUN(wcscoll, (const wchar_t *, const wchar_t *));
wchar_t	*_EXFUN(wcscpy, (wchar_t * , const wchar_t *));
wchar_t	*_EXFUN(wcpcpy, (wchar_t * , const wchar_t *));
size_t	_EXFUN(wcscspn, (const wchar_t *, const wchar_t *));
size_t	_EXFUN(wcslcat, (wchar_t *, const wchar_t *, size_t));
size_t	_EXFUN(wcslcpy, (wchar_t *, const wchar_t *, size_t));
size_t	_EXFUN(wcslen, (const wchar_t *));
wchar_t	*_EXFUN(wcsncat, (wchar_t * , const wchar_t * , size_t));
int	_EXFUN(wcsncmp, (const wchar_t *, const wchar_t *, size_t));
wchar_t	*_EXFUN(wcsncpy, (wchar_t *  , const wchar_t * , size_t));
wchar_t	*_EXFUN(wcpncpy, (wchar_t *  , const wchar_t * , size_t));
size_t	_EXFUN(wcsnlen, (const wchar_t *, size_t));
wchar_t	*_EXFUN(wcspbrk, (const wchar_t *, const wchar_t *));
wchar_t	*_EXFUN(wcsrchr, (const wchar_t *, wchar_t));
size_t	_EXFUN(wcsspn, (const wchar_t *, const wchar_t *));
wchar_t	*_EXFUN(wcsstr, (const wchar_t *, const wchar_t *));
int	_EXFUN(wcswidth, (const wchar_t *, size_t));
size_t	_EXFUN(wcsxfrm, (wchar_t *, const wchar_t *, size_t));
int	_EXFUN(wcwidth, (const wchar_t));
wchar_t	*_EXFUN(wmemchr, (const wchar_t *, wchar_t, size_t));
int	_EXFUN(wmemcmp, (const wchar_t *, const wchar_t *, size_t));
wchar_t	*_EXFUN(wmemcpy, (wchar_t * , const wchar_t * , size_t));
wchar_t	*_EXFUN(wmemmove, (wchar_t *, const wchar_t *, size_t));
wchar_t	*_EXFUN(wmemset, (wchar_t *, wchar_t, size_t));

long    _EXFUN(wcstol, (const wchar_t *, wchar_t **, int));
long long _EXFUN(wcstoll, (const wchar_t *, wchar_t **, int));
unsigned long _EXFUN(wcstoul, (const wchar_t *, wchar_t **, int));
unsigned long long _EXFUN(wcstoull, (const wchar_t *, wchar_t **, int));
long    _EXFUN(_wcstol_r, (struct _reent *, const wchar_t *, wchar_t **, int));
long long _EXFUN(_wcstoll_r, (struct _reent *, const wchar_t *, wchar_t **, int));
unsigned long _EXFUN(_wcstoul_r, (struct _reent *, const wchar_t *, wchar_t **, int));
unsigned long long _EXFUN(_wcstoull_r, (struct _reent *, const wchar_t *, wchar_t **, int));

_END_STD_C

#endif /* !defined(__XC__) */

#endif /* _WCHAR_H_ */
