
#ifndef	_SYS__DEFAULT_FCNTL_H_
#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif
#define	_SYS__DEFAULT_FCNTL_H_
#include <_ansi.h>
#define	_FOPEN		(-1)	/* from sys/file.h, kernel use only */
#define	_FREAD		0x0001	/* read enabled */
#define	_FWRITE		0x0002	/* write enabled */
#define	_FAPPEND	0x0008	/* append (writes guaranteed at the end) */
#define	_FMARK		0x0010	/* internal; mark during gc() */
#define	_FDEFER		0x0020	/* internal; defer for next gc pass */
#define	_FASYNC		0x0040	/* signal pgrp when data ready */
#define	_FSHLOCK	0x0080	/* BSD flock() shared lock present */
#define	_FEXLOCK	0x0100	/* BSD flock() exclusive lock present */
#define	_FCREAT		0x0200	/* open with file create */
#define	_FTRUNC		0x0400	/* open with truncation */
#define	_FEXCL		0x0800	/* error on open if file exists */
#define	_FNBIO		0x1000	/* non blocking I/O (sys5 style) */
#define	_FSYNC		0x2000	/* do all writes synchronously */
#define	_FNONBLOCK	0x4000	/* non blocking I/O (POSIX style) */
#define	_FNDELAY	_FNONBLOCK	/* non blocking I/O (4.2 style) */
#define	_FNOCTTY	0x8000	/* don't assign a ctty on this open */

#define	O_ACCMODE	(O_RDONLY|O_WRONLY|O_RDWR)

/*
 * Flag values for open(2) and fcntl(2)
 * The kernel adds 1 to the open modes to turn it into some
 * combination of FREAD and FWRITE.
 */
#define	O_RDONLY	0		/* +1 == FREAD */
#define	O_WRONLY	1		/* +1 == FWRITE */
#define	O_RDWR		2		/* +1 == FREAD|FWRITE */
#define	O_APPEND	_FAPPEND
#define	O_CREAT		_FCREAT
#define	O_TRUNC		_FTRUNC
#define	O_EXCL		_FEXCL
#define O_SYNC		_FSYNC
/*	O_NDELAY	_FNDELAY 	set in include/fcntl.h */
/*	O_NDELAY	_FNBIO 		set in include/fcntl.h */
#define	O_NONBLOCK	_FNONBLOCK
#define	O_NOCTTY	_FNOCTTY
/* For machines which care - */
#if defined (_WIN32) || defined (__CYGWIN__)
#define _FBINARY        0x10000
#define _FTEXT          0x20000
#define _FNOINHERIT	0x40000

#define O_BINARY	_FBINARY
#define O_TEXT		_FTEXT
#define O_NOINHERIT	_FNOINHERIT

/* The windows header files define versions with a leading underscore.  */
#define _O_RDONLY	O_RDONLY
#define _O_WRONLY	O_WRONLY
#define _O_RDWR		O_RDWR
#define _O_APPEND	O_APPEND
#define _O_CREAT	O_CREAT
#define _O_TRUNC	O_TRUNC
#define _O_EXCL		O_EXCL
#define _O_TEXT		O_TEXT
#define _O_BINARY	O_BINARY
#define _O_RAW		O_BINARY
#define _O_NOINHERIT	O_NOINHERIT
#endif

#ifndef	_POSIX_SOURCE

/*
 * Flags that work for fcntl(fd, F_SETFL, FXXXX)
 */
#define	FAPPEND		_FAPPEND
#define	FSYNC		_FSYNC
#define	FASYNC		_FASYNC
#define	FNBIO		_FNBIO
#define	FNONBIO		_FNONBLOCK	/* XXX fix to be NONBLOCK everywhere */
#define	FNDELAY		_FNDELAY

/*
 * Flags that are disallowed for fcntl's (FCNTLCANT);
 * used for opens, internal state, or locking.
 */
#define	FREAD		_FREAD
#define	FWRITE		_FWRITE
#define	FMARK		_FMARK
#define	FDEFER		_FDEFER
#define	FSHLOCK		_FSHLOCK
#define	FEXLOCK		_FEXLOCK

/*
 * The rest of the flags, used only for opens
 */
#define	FOPEN		_FOPEN
#define	FCREAT		_FCREAT
#define	FTRUNC		_FTRUNC
#define	FEXCL		_FEXCL
#define	FNOCTTY		_FNOCTTY

#endif	/* !_POSIX_SOURCE */

/* XXX close on exec request; must match UF_EXCLOSE in user.h */
#define	FD_CLOEXEC	1	/* posix */

/* fcntl(2) requests */
#define	F_DUPFD		0	/* Duplicate fildes */
#define	F_GETFD		1	/* Get fildes flags (close on exec) */
#define	F_SETFD		2	/* Set fildes flags (close on exec) */
#define	F_GETFL		3	/* Get file flags */
#define	F_SETFL		4	/* Set file flags */
#ifndef	_POSIX_SOURCE
#define	F_GETOWN 	5	/* Get owner - for ASYNC */
#define	F_SETOWN 	6	/* Set owner - for ASYNC */
#endif	/* !_POSIX_SOURCE */
#define	F_GETLK  	7	/* Get record-locking information */
#define	F_SETLK  	8	/* Set or Clear a record-lock (Non-Blocking) */
#define	F_SETLKW 	9	/* Set or Clear a record-lock (Blocking) */
#ifndef	_POSIX_SOURCE
#define	F_RGETLK 	10	/* Test a remote lock to see if it is blocked */
#define	F_RSETLK 	11	/* Set or unlock a remote lock */
#define	F_CNVT 		12	/* Convert a fhandle to an open fd */
#define	F_RSETLKW 	13	/* Set or Clear remote record-lock(Blocking) */
#endif	/* !_POSIX_SOURCE */

/* fcntl(2) flags (l_type field of flock structure) */
#define	F_RDLCK		1	/* read lock */
#define	F_WRLCK		2	/* write lock */
#define	F_UNLCK		3	/* remove lock(s) */
#ifndef	_POSIX_SOURCE
#define	F_UNLKSYS	4	/* remove remote locks for a given system */
#endif	/* !_POSIX_SOURCE */

#ifdef __CYGWIN__
/* Special descriptor value to denote the cwd in calls to openat(2) etc. */
#define AT_FDCWD -2

/* Flag values for faccessat2) et al. */
#define AT_EACCESS              1
#define AT_SYMLINK_NOFOLLOW     2
#define AT_SYMLINK_FOLLOW       4
#define AT_REMOVEDIR            8
#endif

/*#include <sys/stdtypes.h>*/

#ifndef __CYGWIN__
/* file segment locking set data type - information passed to system by user */
struct flock {
	short	l_type;		/* F_RDLCK, F_WRLCK, or F_UNLCK */
	short	l_whence;	/* flag to choose starting offset */
	long	l_start;	/* relative offset, in bytes */
	long	l_len;		/* length, in bytes; 0 means lock to EOF */
	short	l_pid;		/* returned with F_GETLK */
	short	l_xxx;		/* reserved for future use */
};
#endif /* __CYGWIN__ */

#ifndef	_POSIX_SOURCE
/* extended file segment locking set data type */
struct eflock {
	short	l_type;		/* F_RDLCK, F_WRLCK, or F_UNLCK */
	short	l_whence;	/* flag to choose starting offset */
	long	l_start;	/* relative offset, in bytes */
	long	l_len;		/* length, in bytes; 0 means lock to EOF */
	short	l_pid;		/* returned with F_GETLK */
	short	l_xxx;		/* reserved for future use */
	long	l_rpid;		/* Remote process id wanting this lock */
	long	l_rsys;		/* Remote system id wanting this lock */
};
#endif	/* !_POSIX_SOURCE */


#include <sys/types.h>
#include <sys/stat.h>		/* sigh. for the mode bits for open/creat */

extern int open _PARAMS ((const char *, int, ...));
extern int creat _PARAMS ((const char *, mode_t));
extern int fcntl _PARAMS ((int, int, ...));
#ifdef __CYGWIN__
#include <sys/time.h>
extern int futimesat _PARAMS ((int, const char *, const struct timeval *));
extern int openat _PARAMS ((int, const char *, int, ...));
extern int unlinkat _PARAMS ((int, const char *, int));
#endif

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
#endif	/* !_SYS__DEFAULT_FCNTL_H_ */
