#ifndef _CTYPE_H_
#define _CTYPE_H_

#include "_ansi.h"

_BEGIN_STD_C

int _EXFUN(isalnum, (int __c));
int _EXFUN(isalpha, (int __c));
int _EXFUN(iscntrl, (int __c));
int _EXFUN(isdigit, (int __c));
int _EXFUN(isgraph, (int __c));
int _EXFUN(islower, (int __c));
int _EXFUN(isprint, (int __c));
int _EXFUN(ispunct, (int __c));
int _EXFUN(isspace, (int __c));
int _EXFUN(isupper, (int __c));
int _EXFUN(isxdigit,(int __c));
int _EXFUN(tolower, (int __c));
int _EXFUN(toupper, (int __c));

#if !defined(__STRICT_ANSI__) || __cplusplus >= 201103L || __STDC_VERSION__ >= 199901L
int _EXFUN(isblank, (int __c));
#endif
#ifndef __STRICT_ANSI__
int _EXFUN(isascii, (int __c));
int _EXFUN(toascii, (int __c));
int _EXFUN(_tolower, (int __c));
int _EXFUN(_toupper, (int __c));
#endif

#define	_U	01
#define	_L	02
#define	_N	04
#define	_S	010
#define _P	020
#define _C	040
#define _X	0100
#define	_B	0200

extern	__IMPORT _CONST char	*__ctype_ptr__;

/* XC will get the library versions */
#if !defined(__cplusplus) && !defined(__XC__)
#define	isalpha(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&(_U|_L))
#define	isupper(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&_U)
#define	islower(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&_L)
#define	isdigit(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&_N)
#define	isxdigit(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&(_X|_N))
#define	isspace(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&_S)
#define ispunct(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&_P)
#define isalnum(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&(_U|_L|_N))
#define isprint(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&(_P|_U|_L|_N|_B))
#define	isgraph(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&(_P|_U|_L|_N))
#define iscntrl(c)	((__ctype_ptr__)[(unsigned)((c)+1)]&_C)


/* Non-gcc versions will get the library versions, and will be
   slightly slower */
#ifdef __GNUC__
# define toupper(c) \
	__extension__ ({ int __x = (c); islower(__x) ? (__x - 'a' + 'A') : __x;})
# define tolower(c) \
	__extension__ ({ int __x = (c); isupper(__x) ? (__x - 'A' + 'a') : __x;})
#endif
#endif /* !defined(__cplusplus) && !defined(__XC__) */

#ifndef __STRICT_ANSI__
#define isascii(c)	((unsigned)(c)<=0177)
#define toascii(c)	((c)&0177)
#endif

/* For C++ backward-compatibility only.  */
extern	__IMPORT _CONST char	_ctype_[];

_END_STD_C

#endif /* _CTYPE_H_ */
