// -*- C++ -*-
//===--------------------------- cstring ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_CSTRING
#define _LIBCPP_CSTRING

/*
    cstring synopsis

Macros:

    NULL

namespace std
{

Types:

    size_t

void* memcpy(void* restrict s1, const void* restrict s2, size_t n);
void* memmove(void* s1, const void* s2, size_t n);
char* strcpy (char* restrict s1, const char* restrict s2);
char* strncpy(char* restrict s1, const char* restrict s2, size_t n);
char* strcat (char* restrict s1, const char* restrict s2);
char* strncat(char* restrict s1, const char* restrict s2, size_t n);
int memcmp(const void* s1, const void* s2, size_t n);
int strcmp (const char* s1, const char* s2);
int strncmp(const char* s1, const char* s2, size_t n);
int strcoll(const char* s1, const char* s2);
size_t strxfrm(char* restrict s1, const char* restrict s2, size_t n);
const void* memchr(const void* s, int c, size_t n);
      void* memchr(      void* s, int c, size_t n);
const char* strchr(const char* s, int c);
      char* strchr(      char* s, int c);
size_t strcspn(const char* s1, const char* s2);
const char* strpbrk(const char* s1, const char* s2);
      char* strpbrk(      char* s1, const char* s2);
const char* strrchr(const char* s, int c);
      char* strrchr(      char* s, int c);
size_t strspn(const char* s1, const char* s2);
const char* strstr(const char* s1, const char* s2);
      char* strstr(      char* s1, const char* s2);
char* strtok(char* restrict s1, const char* restrict s2);
void* memset(void* s, int c, size_t n);
char* strerror(int errnum);
size_t strlen(const char* s);

}  // std

*/

#include <__config>
#include <string.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

using ::size_t;
using ::memcpy;
using ::memmove;
using ::strcpy;
using ::strncpy;
using ::strcat;
using ::strncat;
using ::memcmp;
using ::strcmp;
using ::strncmp;
using ::strcoll;
using ::strxfrm;

using ::memchr;

using ::strchr;

using ::strcspn;

using ::strpbrk;

using ::strrchr;

using ::strspn;

using ::strstr;

// MSVCRT, GNU libc and its derivates already have the correct prototype in <string.h> #ifdef __cplusplus
#if !defined(__GLIBC__) && !defined(_LIBCPP_MSVCRT) && !defined(__sun__) && !defined(_STRING_H_CPLUSPLUS_98_CONFORMANCE_)
inline _LIBCPP_INLINE_VISIBILITY       char* strchr(      char* __s, int __c) {return ::strchr(__s, __c);}
inline _LIBCPP_INLINE_VISIBILITY       char* strpbrk(      char* __s1, const char* __s2) {return ::strpbrk(__s1, __s2);}
inline _LIBCPP_INLINE_VISIBILITY       char* strrchr(      char* __s, int __c) {return ::strrchr(__s, __c);}
inline _LIBCPP_INLINE_VISIBILITY       void* memchr(      void* __s, int __c, size_t __n) {return ::memchr(__s, __c, __n);}
inline _LIBCPP_INLINE_VISIBILITY       char* strstr(      char* __s1, const char* __s2) {return ::strstr(__s1, __s2);}
#endif

using ::strtok;
using ::memset;
using ::strerror;
using ::strlen;

_LIBCPP_END_NAMESPACE_STD

#endif  // _LIBCPP_CSTRING
