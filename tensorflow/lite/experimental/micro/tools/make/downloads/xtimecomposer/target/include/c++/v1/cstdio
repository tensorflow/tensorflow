// -*- C++ -*-
//===---------------------------- cstdio ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_CSTDIO
#define _LIBCPP_CSTDIO

/*
    cstdio synopsis

Macros:

    BUFSIZ
    EOF
    FILENAME_MAX
    FOPEN_MAX
    L_tmpnam
    NULL
    SEEK_CUR
    SEEK_END
    SEEK_SET
    TMP_MAX
    _IOFBF
    _IOLBF
    _IONBF
    stderr
    stdin
    stdout

namespace std
{

Types:

FILE
fpos_t
size_t

int remove(const char* filename);
int rename(const char* old, const char* new);
FILE* tmpfile(void);
char* tmpnam(char* s);
int fclose(FILE* stream);
int fflush(FILE* stream);
FILE* fopen(const char* restrict filename, const char* restrict mode);
FILE* freopen(const char* restrict filename, const char * restrict mode,
              FILE * restrict stream);
void setbuf(FILE* restrict stream, char* restrict buf);
int setvbuf(FILE* restrict stream, char* restrict buf, int mode, size_t size);
int fprintf(FILE* restrict stream, const char* restrict format, ...);
int fscanf(FILE* restrict stream, const char * restrict format, ...);
int printf(const char* restrict format, ...);
int scanf(const char* restrict format, ...);
int snprintf(char* restrict s, size_t n, const char* restrict format, ...);    // C99
int sprintf(char* restrict s, const char* restrict format, ...);
int sscanf(const char* restrict s, const char* restrict format, ...);
int vfprintf(FILE* restrict stream, const char* restrict format, va_list arg);
int vfscanf(FILE* restrict stream, const char* restrict format, va_list arg);  // C99
int vprintf(const char* restrict format, va_list arg);
int vscanf(const char* restrict format, va_list arg);                          // C99
int vsnprintf(char* restrict s, size_t n, const char* restrict format,         // C99
              va_list arg);
int vsprintf(char* restrict s, const char* restrict format, va_list arg);
int vsscanf(const char* restrict s, const char* restrict format, va_list arg); // C99
int fgetc(FILE* stream);
char* fgets(char* restrict s, int n, FILE* restrict stream);
int fputc(int c, FILE* stream);
int fputs(const char* restrict s, FILE* restrict stream);
int getc(FILE* stream);
int getchar(void);
char* gets(char* s);  // removed in C++14
int putc(int c, FILE* stream);
int putchar(int c);
int puts(const char* s);
int ungetc(int c, FILE* stream);
size_t fread(void* restrict ptr, size_t size, size_t nmemb,
             FILE* restrict stream);
size_t fwrite(const void* restrict ptr, size_t size, size_t nmemb,
              FILE* restrict stream);
int fgetpos(FILE* restrict stream, fpos_t* restrict pos);
int fseek(FILE* stream, long offset, int whence);
int fsetpos(FILE*stream, const fpos_t* pos);
long ftell(FILE* stream);
void rewind(FILE* stream);
void clearerr(FILE* stream);
int feof(FILE* stream);
int ferror(FILE* stream);
void perror(const char* s);

}  // std
*/

#include <__config>
#include <stdio.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

// snprintf
#if defined(_LIBCPP_MSVCRT)
#include "support/win32/support.h"
#endif

#ifdef getc
inline _LIBCPP_INLINE_VISIBILITY int __libcpp_getc(FILE* __stream) {return getc(__stream);}
#undef getc
inline _LIBCPP_INLINE_VISIBILITY int getc(FILE* __stream) {return __libcpp_getc(__stream);}
#endif  // getc

#ifdef putc
inline _LIBCPP_INLINE_VISIBILITY int __libcpp_putc(int __c, FILE* __stream) {return putc(__c, __stream);}
#undef putc
inline _LIBCPP_INLINE_VISIBILITY int putc(int __c, FILE* __stream) {return __libcpp_putc(__c, __stream);}
#endif  // putc

_LIBCPP_BEGIN_NAMESPACE_STD

using ::FILE;
using ::fpos_t;
using ::size_t;

using ::remove;
using ::rename;
using ::tmpfile;
using ::tmpnam;
using ::fclose;
using ::fflush;
using ::fopen;
using ::freopen;
using ::setbuf;
using ::setvbuf;
using ::fprintf;
using ::fscanf;
using ::printf;
using ::scanf;
using ::snprintf;
using ::sprintf;
using ::sscanf;
#ifndef _LIBCPP_MSVCRT
using ::vfprintf;
using ::vfscanf;
using ::vscanf;
using ::vsscanf;
#endif // _LIBCPP_MSVCRT
using ::vprintf;
using ::vsnprintf;
using ::vsprintf;
using ::fgetc;
using ::fgets;
using ::fputc;
using ::fputs;
using ::getc;
using ::getchar;
#if _LIBCPP_STD_VER <= 11
using ::gets;
#endif
using ::putc;
using ::putchar;
using ::puts;
using ::ungetc;
using ::fread;
using ::fwrite;
using ::fgetpos;
using ::fseek;
using ::fsetpos;
using ::ftell;
using ::rewind;
using ::clearerr;
using ::feof;
using ::ferror;
using ::perror;

_LIBCPP_END_NAMESPACE_STD

#endif  // _LIBCPP_CSTDIO
