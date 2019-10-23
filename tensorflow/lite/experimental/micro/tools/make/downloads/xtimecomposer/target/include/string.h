/*
 * string.h
 *
 * Definitions for memory and string functions.
 */

#ifndef _STRING_H_
#define	_STRING_H_

#include "_ansi.h"

#include <sys/reent.h>

#define __need_size_t
#include <stddef.h>

#ifndef NULL
#define NULL 0
#endif

_BEGIN_STD_C

_PTR 	 _EXFUN(memchr,(const _PTR, int, size_t));
int 	 _EXFUN(memcmp,(const _PTR, const _PTR, size_t));
_PTR 	 _EXFUN(memcpy,(_PTR, const _PTR, size_t));
_PTR	 _EXFUN(memmove,(_PTR, const _PTR, size_t));
_PTR	 _EXFUN(memset,(_PTR, int, size_t));
char 	*_EXFUN(strcat,(char *, const char *));
char 	*_EXFUN(strchr,(const char *, int));
int	 _EXFUN(strcmp,(const char *__s1, const char *__s2));
#if !defined(__XC__)
int	 _EXFUN(strcoll,(const char *, const char *));
#endif /* !defined(__XC__) */
char 	*_EXFUN(strcpy,(char *, const char *));
size_t	 _EXFUN(strcspn,(const char *__s1, const char *__s2));
char 	*_EXFUN(strerror,(int));
size_t	 _EXFUN(strlen,(const char *__s));
char 	*_EXFUN(strncat,(char *, const char *, size_t));
int	 _EXFUN(strncmp,(const char *__s1, const char *__s2, size_t));
char 	*_EXFUN(strncpy,(char *, const char *, size_t));
char 	*_EXFUN(strpbrk,(const char *, const char *));
char 	*_EXFUN(strrchr,(const char *, int));
size_t	 _EXFUN(strspn,(const char *__s1, const char *__s2));
char 	*_EXFUN(strstr,(const char *, const char *));
#if !defined(__XC__)
char 	*_EXFUN(strtok,(char *, const char *));
#endif /* !defined(__XC__) */

#if !defined(__XC__)
size_t	 _EXFUN(strxfrm,(char *, const char *, size_t));
#endif /* !defined(__XC__) */

#ifndef __STRICT_ANSI__
char 	*_EXFUN(strtok_r,(char *, const char *, char **));

int	 _EXFUN(bcmp,(const void *, const void *, size_t));
void	 _EXFUN(bcopy,(const void *, void *, size_t));
void	 _EXFUN(bzero,(void *, size_t));
int	 _EXFUN(ffs,(int));
char 	*_EXFUN(index,(const char *, int));
_PTR	 _EXFUN(memccpy,(_PTR, const _PTR, int, size_t));
_PTR	 _EXFUN(mempcpy,(_PTR, const _PTR, size_t));
_PTR	 _EXFUN(memmem, (const _PTR, size_t, const _PTR, size_t));
char 	*_EXFUN(rindex,(const char *, int));
char 	*_EXFUN(stpcpy,(char *, const char *));
char 	*_EXFUN(stpncpy,(char *, const char *, size_t));
int	 _EXFUN(strcasecmp,(const char *, const char *));
char	*_EXFUN(strcasestr,(const char *, const char *));
char 	*_EXFUN(strdup,(const char *));
char 	*_EXFUN(strndup,(const char *, size_t));
char 	*_EXFUN(strerror_r,(int, char *, size_t));
size_t	 _EXFUN(strlcat,(char *, const char *, size_t));
size_t	 _EXFUN(strlcpy,(char *, const char *, size_t));
int	 _EXFUN(strncasecmp,(const char *, const char *, size_t));
size_t	 _EXFUN(strnlen,(const char *, size_t));
char 	*_EXFUN(strsep,(char **, const char *));
char	*_EXFUN(strlwr,(char *));
char	*_EXFUN(strupr,(char *));
#ifdef __CYGWIN__
#ifndef DEFS_H	/* Kludge to work around problem compiling in gdb */
char  *_EXFUN(strsignal, (int __signo));
#endif
int     _EXFUN(strtosigno, (const char *__name));
#endif

/* These function names are used on Windows and perhaps other systems.  */
#ifndef strcmpi
#define strcmpi strcasecmp
#endif
#ifndef stricmp
#define stricmp strcasecmp
#endif
#ifndef strncmpi
#define strncmpi strncasecmp
#endif
#ifndef strnicmp
#define strnicmp strncasecmp
#endif

#endif /* ! __STRICT_ANSI__ */

#include <sys/string.h>

_END_STD_C

#endif /* _STRING_H_ */
