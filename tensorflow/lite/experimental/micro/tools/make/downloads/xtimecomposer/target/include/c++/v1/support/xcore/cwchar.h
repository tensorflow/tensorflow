#ifndef XCORE_CWCHAR_H_INCLUDED
#define XCORE_CWCHAR_H_INCLUDED

#ifdef __cplusplus
extern "C"  {
#endif


// Clang handles 'wchar_t' as a 'char', value range 0-255.

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

static inline int fwprintf(FILE* __restrict stream, const wchar_t* __restrict format, ...)
  {va_list arg;va_start(arg,format);int r=vfprintf(stream,(const char*)format,arg);va_end(arg);return r;}
static inline int fwscanf(FILE* __restrict stream, const wchar_t* __restrict format, ...)
  {va_list arg;va_start(arg,format);int r=vfscanf(stream,(const char*)format,arg);va_end(arg);return r;}
static inline int swprintf(wchar_t* __restrict s, size_t n, const wchar_t* __restrict format, ...)
  {va_list arg;va_start(arg,format);int r=vsnprintf((char*)s,n,(const char*)format,arg);va_end(arg);return r;}
static inline int swscanf(const wchar_t* __restrict s, const wchar_t* __restrict format, ...)
  {va_list arg;va_start(arg,format);int r=vsscanf((const char*)s,(const char*)format,arg);va_end(arg);return r;}
static inline int vfwprintf(FILE* __restrict stream, const wchar_t* __restrict format, va_list arg)
  {return vfprintf(stream,(const char*)format,arg);}
static inline int vfwscanf(FILE* __restrict stream, const wchar_t* __restrict format, va_list arg)
  {return vfscanf(stream,(const char*)format,arg);}
static inline int vswprintf(wchar_t* __restrict s, size_t n, const wchar_t* __restrict format, va_list arg)
  {return vsnprintf((char*)s,n,(const char*)format,arg);}
static inline int vswscanf(const wchar_t* __restrict s, const wchar_t* __restrict format, va_list arg)
  {return vsscanf((const char*)s,(const char*)format,arg);}
static inline int vwprintf(const wchar_t* __restrict format, va_list arg)
  {return vprintf((const char*)format,arg);}
static inline int vwscanf(const wchar_t* __restrict format, va_list arg)
  {return vscanf((const char*)format,arg);}
static inline int wprintf(const wchar_t* __restrict format, ...)
  {va_list arg;va_start(arg,format);int r=vprintf((const char*)format,arg);va_end(arg);return r;}
static inline int wscanf(const wchar_t* __restrict format, ...)
  {va_list arg;va_start(arg,format);int r=vscanf((const char*)format,arg);va_end(arg);return r;}

static inline wint_t fgetwc(FILE* __restrict stream)
  {return (wint_t)fgetc(stream);}
static inline wchar_t* fgetws(wchar_t* __restrict s, int n, FILE* __restrict stream)
  {return (wchar_t*)fgets((char*)s,n,stream);}
static inline wint_t fputwc(wchar_t c, FILE* __restrict stream)
  {return (wint_t)fputc((char)c,stream);}
static inline int fputws(const wchar_t* __restrict s, FILE* __restrict stream)
  {return fputs((const char*)s,stream);}
static inline int fwide(FILE* __restrict stream, int mode)
  {(void)stream; (void)mode; return 0;}; // we can't return both wide and byte!
static inline wint_t getwc(FILE* __restrict stream)
  {return (wint_t)getc(stream);}
static inline wint_t getwchar()
  {return (wint_t)getchar();}
static inline wint_t putwc(wchar_t c, FILE* __restrict stream)
  {return (wint_t)putc((char)c,stream);}
static inline wint_t putwchar(wchar_t c)
  {return (wint_t)putchar((char)c);}
static inline wint_t ungetwc(wint_t c, FILE* __restrict stream)
  {return (wint_t)ungetc((char)c,stream);}
static inline double wcstod(const wchar_t* nptr, wchar_t** endptr)
  {return strtod((const char*)nptr,(char**)endptr);}
static inline float wcstof(const wchar_t* nptr, wchar_t** endptr)
  {return strtof((const char*)nptr,(char**)endptr);}
static inline long double wcstold(const wchar_t* nptr, wchar_t** endptr)
  {return strtod((const char*)nptr,(char**)endptr);}

extern char* strtok_r(char *, const char *, char **); // not declared if __STRICT_ANSI__
static inline wchar_t* wcstok(wchar_t* s1, const wchar_t* s2, wchar_t** ptr)
  {return (wchar_t*)strtok_r((char*)s1,(const char*)s2, (char**)ptr);}
static inline size_t wcsftime(wchar_t* __restrict s, size_t maxsize, const wchar_t* __restrict format, const struct tm* timeptr)
  {return strftime((char*)s,maxsize,(const char*)format,timeptr);}


#ifdef __cplusplus
}
#endif

#endif // XCORE_CWCHAR_H_INCLUDED
