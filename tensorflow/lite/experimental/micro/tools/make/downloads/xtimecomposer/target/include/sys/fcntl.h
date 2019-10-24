/* This file is to be kept in sync with newlib/libc/include/sys/fcntl.h,
   on which it is based, except values used or returned by syscalls must
   be those.  */
#ifndef	_FCNTL_
#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif
#define	_FCNTL_
#include <_ansi.h>

#define	O_ACCMODE	(O_RDONLY|O_WRONLY|O_RDWR)

#define	O_RDONLY	0x0001		/* +1 == FREAD */
#define	O_WRONLY	0x0002		/* +1 == FWRITE */
#define	O_RDWR		0x0004		/* +1 == FREAD|FWRITE */
#define	O_CREAT		0x0100
#define	O_TRUNC		0x0200
#define	O_EXCL		0x0400
#define	O_APPEND	0x0800
/*                0x1000 reserved */
#define O_BINARY	0x8000

#include <sys/types.h>
#include <sys/stat.h>		/* sigh. for the mode bits for open/creat */

extern int open _PARAMS ((const char *, int, ...));
extern int creat _PARAMS ((const char *, mode_t));
extern int fcntl _PARAMS ((int, int, ...));

/* Provide _<systemcall> prototypes for functions provided by some versions
   of newlib.  */
#ifdef _COMPILING_NEWLIB
extern int _open _PARAMS ((const char *, int, ...));
extern int _fcntl _PARAMS ((int, int, ...));
#ifdef __LARGE64_FILES
extern int _open64 _PARAMS ((const char *, int, ...));
#endif
#endif

#if defined(__cplusplus) || defined(__XC__)
}
#endif
#endif	/* !_FCNTL_ */
