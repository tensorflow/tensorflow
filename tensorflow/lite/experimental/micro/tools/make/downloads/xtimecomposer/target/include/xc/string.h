#ifndef _xc_string_h_
#define _xc_string_h_

#include_next <string.h>
#include <safe/string.h>

#if defined(__XC__)
#if defined(UNSAFE_LIBC)
#define memcpy(s1, s2, n) __builtin_memcpy_xc(s1, s2, n)
#else
// Should you have trouble with movable pointers being aliased, use safestring.h instead.
#define memchr(s, c, n) _safe_memchr((const char *)s, c, n)
#define memcmp(s1, s2, n) _safe_memcmp((const char *)s1, (const char *)s2, n)
#define memcpy(s1, s2, n) _safe_memcpy(s1, s2, n)
#define memmove(s1, s2, n) _safe_memmove((char *)s1, (const char *)s2, n)
#define memset(s, c, n) _safe_memset((char *)s, c, n)
#define strcat(s1, s2) _safe_strcat(s1, s2)
#define strchr(s, c) _safe_strchr(s, c)
#define strcmp(s1, s2) _safe_strcmp(s1, s2)
#define strcpy(s1, s2) _safe_strcpy(s1, s2)
#define strcspn(s1, s2) _safe_strcspn(s1, s2)
#define strerror(errnum) _safe_strerror(errnum)
#define strlen(s) _safe_strlen(s)
#define strncat(s1, s2, n) _safe_strncat(s1, s2, n)
#define strncmp(s1, s2, n) _safe_strncmp(s1, s2, n)
#define strncpy(s1, s2, n) _safe_strncpy(s1, s2, n)
#define strpbrk(s1, s2) _safe_strpbrk(s1, s2)
#define strrchr(s, c) _safe_strrchr(s, c)
#define strspn(s1, s2) _safe_strspn(s1, s2)
#define strstr(s1, s2) _safe_strstr(s1, s2)
#define strnlen(s, n) _safe_strnlen(s, n)
#endif /* UNSAFE_LIBC */
#endif /* __XC__ */

#endif /* _xc_string_h_ */
