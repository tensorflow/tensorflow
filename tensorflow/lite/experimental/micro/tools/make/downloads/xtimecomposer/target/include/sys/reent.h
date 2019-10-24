/* This header file provides the reentrancy.  */

/* WARNING: All identifiers here must begin with an underscore.  This file is
   included by stdio.h and others and we therefore must only use identifiers
   in the namespace allotted to us.  */

#ifndef _SYS_REENT_H_
#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif
#define _SYS_REENT_H_

#include <_ansi.h>
#include <sys/_types.h>

#define _NULL 0

#ifndef __Long
#if __LONG_MAX__ == 2147483647L
#define __Long long
typedef unsigned __Long __ULong;
#elif __INT_MAX__ == 2147483647
#define __Long int
typedef unsigned __Long __ULong;
#endif
#endif

#if !defined( __Long)
#include <sys/types.h>
#endif

#ifndef __Long
#define __Long __int32_t
typedef __uint32_t __ULong;
#endif
struct _reent;

/*
 * Stdio buffers.
 *
 * This and __FILE are defined here because we need them for struct _reent,
 * but we don't want stdio.h included when stdlib.h is.
 */

struct __sbuf {
	unsigned char *_base;
	int	_size;
};

/*
 * Stdio state variables.
 *
 * The following always hold:
 *
 *	if (_flags&(__SLBF|__SWR)) == (__SLBF|__SWR),
 *		_lbfsize is -_bf._size, else _lbfsize is 0
 *	if _flags&__SRD, _w is 0
 *	if _flags&__SWR, _r is 0
 *
 * This ensures that the getc and putc macros (or inline functions) never
 * try to write or read from a file that is in `read' or `write' mode.
 * (Moreover, they can, and do, automatically switch from read mode to
 * write mode, and back, on "r+" and "w+" files.)
 *
 * _lbfsize is used only to make the inline line-buffered output stream
 * code as compact as possible.
 *
 * _ub, _up, and _ur are used when ungetc() pushes back more characters
 * than fit in the current _bf, or when ungetc() pushes back a character
 * that does not match the previous one in _bf.  When this happens,
 * _ub._base becomes non-nil (i.e., a stream has ungetc() data iff
 * _ub._base!=NULL) and _up and _ur save the current values of _p and _r.
 */

#ifdef __XC__
struct __sFILE;
#else
struct __sFILE {
  unsigned char *_p;	/* current position in (some) buffer */
  int	_r;		/* read space left for getc() */
  int	_w;		/* write space left for putc() */
  short	_flags;		/* flags, below; this FILE is free if 0 */
  short	_file;		/* fileno, if Unix descriptor, else -1 */
  struct __sbuf _bf;	/* the buffer (at least 1 byte, if !NULL) */
  int	_lbfsize;	/* 0 or -_bf._size, for inline putc */

  /* operations */
  _PTR	_cookie;	/* cookie passed to io functions */

  __attribute__((fptrgroup("__sread",0))) // Don't call __checkFptrGroup
  _READ_WRITE_RETURN_TYPE _EXFUN((*_read),(struct _reent *, _PTR,
					   char *, int));
  __attribute__((fptrgroup("__swrite",0))) // Don't call __checkFptrGroup
  _READ_WRITE_RETURN_TYPE _EXFUN((*_write),(struct _reent *, _PTR,
					    const char *, int));
  __attribute__((fptrgroup("__sseek",0))) // Don't call __checkFptrGroup
  _fpos_t _EXFUN((*_seek),(struct _reent *, _PTR, _fpos_t, int));
  __attribute__((fptrgroup("__sclose",0))) // Don't call __checkFptrGroup
  int _EXFUN((*_close),(struct _reent *, _PTR));

  /* separate buffer for long sequences of ungetc() */
  struct __sbuf _ub;	/* ungetc buffer */
  unsigned char *_up;	/* saved _p when _p is doing ungetc data */
  int	_ur;		/* saved _r when _r is counting ungetc data */

  /* tricks to meet minimum requirements even when malloc() fails */
  unsigned char _ubuf[3];	/* guarantee an ungetc() buffer */
  unsigned char _nbuf[1];	/* guarantee a getc() buffer */

  /* separate buffer for fgetline() when line crosses buffer boundary */
  struct __sbuf _lb;	/* buffer for fgetline() */

  /* Unix stdio files get aligned to block boundaries on fseek() */
  int	_blksize;	/* stat.st_blksize (may be != _bf._size) */
  int	_offset;	/* current lseek offset */

#ifndef __SINGLE_THREAD__
  _flock_t _lock;	/* for thread-safety locking */
#endif
};
#endif /* __XC__ */

#ifdef __CUSTOM_FILE_IO__

/* Get custom _FILE definition.  */
#include <sys/custom_file.h>

#else /* !__CUSTOM_FILE_IO__ */
#ifdef __LARGE64_FILES
struct __sFILE64 {
  unsigned char *_p;	/* current position in (some) buffer */
  int	_r;		/* read space left for getc() */
  int	_w;		/* write space left for putc() */
  short	_flags;		/* flags, below; this FILE is free if 0 */
  short	_file;		/* fileno, if Unix descriptor, else -1 */
  struct __sbuf _bf;	/* the buffer (at least 1 byte, if !NULL) */
  int	_lbfsize;	/* 0 or -_bf._size, for inline putc */

  struct _reent *_data;

  /* operations */
  _PTR	_cookie;	/* cookie passed to io functions */

  _READ_WRITE_RETURN_TYPE _EXFUN((*_read),(struct _reent *, _PTR,
					   char *, int));
  _READ_WRITE_RETURN_TYPE _EXFUN((*_write),(struct _reent *, _PTR,
					    const char *, int));
  _fpos_t _EXFUN((*_seek),(struct _reent *, _PTR, _fpos_t, int));
  int _EXFUN((*_close),(struct _reent *, _PTR));

  /* separate buffer for long sequences of ungetc() */
  struct __sbuf _ub;	/* ungetc buffer */
  unsigned char *_up;	/* saved _p when _p is doing ungetc data */
  int	_ur;		/* saved _r when _r is counting ungetc data */

  /* tricks to meet minimum requirements even when malloc() fails */
  unsigned char _ubuf[3];	/* guarantee an ungetc() buffer */
  unsigned char _nbuf[1];	/* guarantee a getc() buffer */

  /* separate buffer for fgetline() when line crosses buffer boundary */
  struct __sbuf _lb;	/* buffer for fgetline() */

  /* Unix stdio files get aligned to block boundaries on fseek() */
  int	_blksize;	/* stat.st_blksize (may be != _bf._size) */
  int   _flags2;        /* for future use */

  _off64_t _offset;     /* current lseek offset */
  _fpos64_t _EXFUN((*_seek64),(struct _reent *, _PTR, _fpos64_t, int));

#ifndef __SINGLE_THREAD__
  _flock_t _lock;	/* for thread-safety locking */
#endif
};
typedef struct __sFILE64 __FILE;
#else
typedef struct __sFILE   __FILE;
#endif /* __LARGE64_FILES */
#endif /* !__CUSTOM_FILE_IO__ */

struct _glue 
{
  struct _glue *_next;
  int _niobs;
  __FILE *_iobs;
};

/*
 * struct _reent
 *
 * No longer used.
 */

struct _reent;

extern _VOID   _EXFUN(_cleanup,(_VOID));

extern __FILE *__stdin, *__stdout, *__stderr;

__FILE * _EXFUN(__getstdin, (void));
__FILE * _EXFUN(__getstdout, (void));
__FILE * _EXFUN(__getstderr, (void));

#define _REENT _NULL
#if defined(__cplusplus) || defined(__XC__)
}
#endif
#endif /* _SYS_REENT_H_ */
