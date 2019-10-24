// -*- C++ -*-
//===------------------- support/xlocale/xlocale.h ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is a shared implementation of a shim to provide extended locale support
// on top of libc's that don't support it (like Android's bionic, and Newlib).
//
// The 'illusion' only works when the specified locale is "C" or "POSIX", but
// that's about as good as we can do without implementing full xlocale support
// in the underlying libc.
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_XLOCALE_XLOCALE_H
#define _LIBCPP_SUPPORT_XLOCALE_XLOCALE_H

#ifdef __cplusplus
extern "C" {
#endif

static inline int isalnum_l(int c, locale_t) {
  return isalnum(c);
}

static inline int isalpha_l(int c, locale_t) {
  return isalpha(c);
}

static inline int isblank_l(int c, locale_t) {
  return isblank(c);
}

static inline int iscntrl_l(int c, locale_t) {
  return iscntrl(c);
}

static inline int isdigit_l(int c, locale_t) {
  return isdigit(c);
}

static inline int isgraph_l(int c, locale_t) {
  return isgraph(c);
}

static inline int islower_l(int c, locale_t) {
  return islower(c);
}

static inline int isprint_l(int c, locale_t) {
  return isprint(c);
}

static inline int ispunct_l(int c, locale_t) {
  return ispunct(c);
}

static inline int isspace_l(int c, locale_t) {
  return isspace(c);
}

static inline int isupper_l(int c, locale_t) {
  return isupper(c);
}

static inline int isxdigit_l(int c, locale_t) {
  return isxdigit(c);
}

static inline int iswalnum_l(wint_t c, locale_t) {
  return iswalnum(c);
}

static inline int iswalpha_l(wint_t c, locale_t) {
  return iswalpha(c);
}

static inline int iswblank_l(wint_t c, locale_t) {
  return iswblank(c);
}

static inline int iswcntrl_l(wint_t c, locale_t) {
  return iswcntrl(c);
}

static inline int iswdigit_l(wint_t c, locale_t) {
  return iswdigit(c);
}

static inline int iswgraph_l(wint_t c, locale_t) {
  return iswgraph(c);
}

static inline int iswlower_l(wint_t c, locale_t) {
  return iswlower(c);
}

static inline int iswprint_l(wint_t c, locale_t) {
  return iswprint(c);
}

static inline int iswpunct_l(wint_t c, locale_t) {
  return iswpunct(c);
}

static inline int iswspace_l(wint_t c, locale_t) {
  return iswspace(c);
}

static inline int iswupper_l(wint_t c, locale_t) {
  return iswupper(c);
}

static inline int iswxdigit_l(wint_t c, locale_t) {
  return iswxdigit(c);
}

static inline int toupper_l(int c, locale_t) {
  return toupper(c);
}

static inline int tolower_l(int c, locale_t) {
  return tolower(c);
}

static inline int towupper_l(int c, locale_t) {
  return towupper(c);
}

static inline int towlower_l(int c, locale_t) {
  return towlower(c);
}

static inline int strcoll_l(const char *s1, const char *s2, locale_t) {
  return strcoll(s1, s2);
}

static inline size_t strxfrm_l(char *dest, const char *src, size_t n,
                               locale_t) {
  return strxfrm(dest, src, n);
}

static inline size_t strftime_l(char *s, size_t max, const char *format,
                                const struct tm *tm, locale_t) {
  return strftime(s, max, format, tm);
}

static inline int wcscoll_l(const wchar_t *ws1, const wchar_t *ws2, locale_t) {
  return wcscoll(ws1, ws2);
}

static inline size_t wcsxfrm_l(wchar_t *dest, const wchar_t *src, size_t n,
                               locale_t) {
  return wcsxfrm(dest, src, n);
}

static inline long double strtold_l(const char *nptr, char **endptr, locale_t) {
  return strtold(nptr, endptr);
}

static inline long long strtoll_l(const char *nptr, char **endptr, int base,
                                  locale_t) {
  return strtoll(nptr, endptr, base);
}

static inline unsigned long long strtoull_l(const char *nptr, char **endptr,
                                            int base, locale_t) {
  return strtoull(nptr, endptr, base);
}

static inline long long wcstoll_l(const wchar_t *nptr, wchar_t **endptr,
                                  int base, locale_t) {
  return wcstoll(nptr, endptr, base);
}

static inline unsigned long long wcstoull_l(const wchar_t *nptr,
                                            wchar_t **endptr, int base,
                                            locale_t) {
  return wcstoull(nptr, endptr, base);
}

static inline long double wcstold_l(const wchar_t *nptr, wchar_t **endptr,
                                    locale_t) {
  return wcstold(nptr, endptr);
}

#ifdef __cplusplus
}
#endif

#endif // _LIBCPP_SUPPORT_XLOCALE_XLOCALE_H
