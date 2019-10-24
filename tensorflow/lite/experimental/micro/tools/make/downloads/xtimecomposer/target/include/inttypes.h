/*
 * Copyright (c) 2004, 2005 by
 * Ralf Corsepius, Ulm/Germany. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this software
 * is freely granted, provided that this notice is preserved.
 */

/**
 *  @file  inttypes.h
 */

#ifndef _INTTYPES_H
#define _INTTYPES_H

#include "_ansi.h"
#include <stdint.h>
#define __need_wchar_t
#include <stddef.h>

#define __STRINGIFY(a) #a

/* 8-bit types */
#define __PRI8(x) __STRINGIFY(x)
#define __SCN8(x) __STRINGIFY(hh##x)
#if __STDINT_EXP(INT_MAX) >= 0x7f
#define __PRIFAST8(x) __STRINGIFY(x)
#define __SCNFAST8(x) __STRINGIFY(x)
#else
#define __PRIFAST8(x) __PRI8(x)
#define __SCNFAST8(x) __SCN8(x)
#endif


#define PRId8		__PRI8(d)
#define PRIi8		__PRI8(i)
#define PRIo8		__PRI8(o)
#define PRIu8		__PRI8(u)
#define PRIx8		__PRI8(x)
#define PRIX8		__PRI8(X)

#define SCNd8		__SCN8(d)
#define SCNi8		__SCN8(i)
#define SCNo8		__SCN8(o)
#define SCNu8		__SCN8(u)
#define SCNx8		__SCN8(x)


#define PRIdLEAST8	__PRI8(d)
#define PRIiLEAST8	__PRI8(i)
#define PRIoLEAST8	__PRI8(o)
#define PRIuLEAST8	__PRI8(u)
#define PRIxLEAST8	__PRI8(x)
#define PRIXLEAST8	__PRI8(X)

#define SCNdLEAST8	__SCN8(d)
#define SCNiLEAST8	__SCN8(i)
#define SCNoLEAST8	__SCN8(o)
#define SCNuLEAST8	__SCN8(u)
#define SCNxLEAST8	__SCN8(x)


#define PRIdFAST8	__PRIFAST8(d)
#define PRIiFAST8	__PRIFAST8(i)
#define PRIoFAST8	__PRIFAST8(o)
#define PRIuFAST8	__PRIFAST8(u)
#define PRIxFAST8	__PRIFAST8(x)
#define PRIXFAST8	__PRIFAST8(X)

#define SCNdFAST8	__SCNFAST8(d)
#define SCNiFAST8	__SCNFAST8(i)
#define SCNoFAST8	__SCNFAST8(o)
#define SCNuFAST8	__SCNFAST8(u)
#define SCNxFAST8	__SCNFAST8(x)

/* 16-bit types */
#define __PRI16(x) __STRINGIFY(x)
#define __SCN16(x) __STRINGIFY(h##x)
#if __STDINT_EXP(INT_MAX) >= 0x7fff
#define __PRIFAST16(x) __STRINGIFY(x)
#define __SCNFAST16(x) __STRINGIFY(x)
#else
#define __PRIFAST16(x) __PRI16(x)
#define __SCNFAST16(x) __SCN16(x)
#endif


#define PRId16		__PRI16(d)
#define PRIi16		__PRI16(i)
#define PRIo16		__PRI16(o)
#define PRIu16		__PRI16(u)
#define PRIx16		__PRI16(x)
#define PRIX16		__PRI16(X)

#define SCNd16		__SCN16(d)
#define SCNi16		__SCN16(i)
#define SCNo16		__SCN16(o)
#define SCNu16		__SCN16(u)
#define SCNx16		__SCN16(x)


#define PRIdLEAST16	__PRI16(d)
#define PRIiLEAST16	__PRI16(i)
#define PRIoLEAST16	__PRI16(o)
#define PRIuLEAST16	__PRI16(u)
#define PRIxLEAST16	__PRI16(x)
#define PRIXLEAST16	__PRI16(X)

#define SCNdLEAST16	__SCN16(d)
#define SCNiLEAST16	__SCN16(i)
#define SCNoLEAST16	__SCN16(o)
#define SCNuLEAST16	__SCN16(u)
#define SCNxLEAST16	__SCN16(x)


#define PRIdFAST16	__PRIFAST16(d)
#define PRIiFAST16	__PRIFAST16(i)
#define PRIoFAST16	__PRIFAST16(o)
#define PRIuFAST16	__PRIFAST16(u)
#define PRIxFAST16	__PRIFAST16(x)
#define PRIXFAST16	__PRIFAST16(X)

#define SCNdFAST16	__SCNFAST16(d)
#define SCNiFAST16	__SCNFAST16(i)
#define SCNoFAST16	__SCNFAST16(o)
#define SCNuFAST16	__SCNFAST16(u)
#define SCNxFAST16	__SCNFAST16(x)

/* 32-bit types */
#if __have_long32
#define __PRI32(x) __STRINGIFY(l##x)
#define __SCN32(x) __STRINGIFY(l##x)
#else
#define __PRI32(x) __STRINGIFY(x)
#define __SCN32(x) __STRINGIFY(x)
#endif
#if __STDINT_EXP(INT_MAX) >= 0x7fffffff
#define __PRIFAST32(x) __STRINGIFY(x)
#define __SCNFAST32(x) __STRINGIFY(x)
#else
#define __PRIFAST32(x) __PRI32(x)
#define __SCNFAST32(x) __SCN32(x)
#endif

#define PRId32		__PRI32(d)
#define PRIi32		__PRI32(i)
#define PRIo32		__PRI32(o)
#define PRIu32		__PRI32(u)
#define PRIx32		__PRI32(x)
#define PRIX32		__PRI32(X)

#define SCNd32		__SCN32(d)
#define SCNi32		__SCN32(i)
#define SCNo32		__SCN32(o)
#define SCNu32		__SCN32(u)
#define SCNx32		__SCN32(x)


#define PRIdLEAST32	__PRI32(d)
#define PRIiLEAST32	__PRI32(i)
#define PRIoLEAST32	__PRI32(o)
#define PRIuLEAST32	__PRI32(u)
#define PRIxLEAST32	__PRI32(x)
#define PRIXLEAST32	__PRI32(X)

#define SCNdLEAST32	__SCN32(d)
#define SCNiLEAST32	__SCN32(i)
#define SCNoLEAST32	__SCN32(o)
#define SCNuLEAST32	__SCN32(u)
#define SCNxLEAST32	__SCN32(x)


#define PRIdFAST32	__PRIFAST32(d)
#define PRIiFAST32	__PRIFAST32(i)
#define PRIoFAST32	__PRIFAST32(o)
#define PRIuFAST32	__PRIFAST32(u)
#define PRIxFAST32	__PRIFAST32(x)
#define PRIXFAST32	__PRIFAST32(X)

#define SCNdFAST32	__SCNFAST32(d)
#define SCNiFAST32	__SCNFAST32(i)
#define SCNoFAST32	__SCNFAST32(o)
#define SCNuFAST32	__SCNFAST32(u)
#define SCNxFAST32	__SCNFAST32(x)


/* 64-bit types */
#if __have_long64
#define __PRI64(x) __STRINGIFY(l##x)
#define __SCN64(x) __STRINGIFY(l##x)
#elif __have_longlong64
#define __PRI64(x) __STRINGIFY(ll##x)
#define __SCN64(x) __STRINGIFY(ll##x)
#else
#define __PRI64(x) __STRINGIFY(x)
#define __SCN64(x) __STRINGIFY(x)
#endif
#if __STDINT_EXP(INT_MAX) > 0x7fffffff
#define __PRIFAST64(x) __STRINGIFY(x)
#define __SCNFAST64(x) __STRINGIFY(x)
#else
#define __PRIFAST64(x) __PRI64(x)
#define __SCNFAST64(x) __SCN64(x)
#endif

#define PRId64		__PRI64(d)
#define PRIi64		__PRI64(i)
#define PRIo64		__PRI64(o)
#define PRIu64		__PRI64(u)
#define PRIx64		__PRI64(x)
#define PRIX64		__PRI64(X)

#define SCNd64		__SCN64(d)
#define SCNi64		__SCN64(i)
#define SCNo64		__SCN64(o)
#define SCNu64		__SCN64(u)
#define SCNx64		__SCN64(x)

#if __int64_t_defined
#define PRIdLEAST64	__PRI64(d)
#define PRIiLEAST64	__PRI64(i)
#define PRIoLEAST64	__PRI64(o)
#define PRIuLEAST64	__PRI64(u)
#define PRIxLEAST64	__PRI64(x)
#define PRIXLEAST64	__PRI64(X)

#define SCNdLEAST64	__SCN64(d)
#define SCNiLEAST64	__SCN64(i)
#define SCNoLEAST64	__SCN64(o)
#define SCNuLEAST64	__SCN64(u)
#define SCNxLEAST64	__SCN64(x)


#define PRIdFAST64	__PRIFAST64(d)
#define PRIiFAST64	__PRIFAST64(i)
#define PRIoFAST64	__PRIFAST64(o)
#define PRIuFAST64	__PRIFAST64(u)
#define PRIxFAST64	__PRIFAST64(x)
#define PRIXFAST64	__PRIFAST64(X)

#define SCNdFAST64	__SCNFAST64(d)
#define SCNiFAST64	__SCNFAST64(i)
#define SCNoFAST64	__SCNFAST64(o)
#define SCNuFAST64	__SCNFAST64(u)
#define SCNxFAST64	__SCNFAST64(x)
#endif

/* max-bit types */
#if __have_long64
#define __PRIMAX(x) __STRINGIFY(l##x)
#define __SCNMAX(x) __STRINGIFY(l##x)
#elif __have_longlong64
#define __PRIMAX(x) __STRINGIFY(ll##x)
#define __SCNMAX(x) __STRINGIFY(ll##x)
#else
#define __PRIMAX(x) __STRINGIFY(x)
#define __SCNMAX(x) __STRINGIFY(x)
#endif

#define PRIdMAX		__PRIMAX(d)
#define PRIiMAX		__PRIMAX(i)
#define PRIoMAX		__PRIMAX(o)
#define PRIuMAX		__PRIMAX(u)
#define PRIxMAX		__PRIMAX(x)
#define PRIXMAX		__PRIMAX(X)

#define SCNdMAX		__SCNMAX(d)
#define SCNiMAX		__SCNMAX(i)
#define SCNoMAX		__SCNMAX(o)
#define SCNuMAX		__SCNMAX(u)
#define SCNxMAX		__SCNMAX(x)

/* ptr types */
#if defined(__xcore__)
#define __PRIPTR(x) __STRINGIFY(x)
#define __SCNPTR(x) __STRINGIFY(x)
#else
#define __PRIPTR(x) __STRINGIFY(l##x)
#define __SCNPTR(x) __STRINGIFY(l##x)
#endif

#define PRIdPTR		__PRIPTR(d)
#define PRIiPTR		__PRIPTR(i)
#define PRIoPTR		__PRIPTR(o)
#define PRIuPTR		__PRIPTR(u)
#define PRIxPTR		__PRIPTR(x)
#define PRIXPTR		__PRIPTR(X)

#define SCNdPTR		__SCNPTR(d)
#define SCNiPTR		__SCNPTR(i)
#define SCNoPTR		__SCNPTR(o)
#define SCNuPTR		__SCNPTR(u)
#define SCNxPTR		__SCNPTR(x)


typedef struct {
  intmax_t	quot;
  intmax_t	rem;
} imaxdiv_t;

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

extern intmax_t  imaxabs(intmax_t j);
extern imaxdiv_t imaxdiv(intmax_t numer, intmax_t denomer);
extern intmax_t  strtoimax(const char * _RESTRICT, char ** _RESTRICT, int);
extern uintmax_t strtoumax(const char * _RESTRICT, char ** _RESTRICT, int);
#if !defined(__XC__)
extern intmax_t  wcstoimax(const wchar_t * _RESTRICT, wchar_t ** _RESTRICT, int);
extern uintmax_t wcstoumax(const wchar_t * _RESTRICT, wchar_t ** _RESTRICT, int);
#endif /* !defined(__XC__) */

#if defined(__cplusplus) || defined(__XC__)
}
#endif

#endif
