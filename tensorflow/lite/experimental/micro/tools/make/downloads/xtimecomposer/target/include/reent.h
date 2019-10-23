/* This header file provides the reentrancy.  */

/* The reentrant system calls here serve two purposes:

   1) Provide reentrant versions of the system calls the ANSI C library
      requires.
   2) Provide these system calls in a namespace clean way.

   It is intended that *all* system calls that the ANSI C library needs
   be declared here.  It documents them all in one place.  All library access
   to the system is via some form of these functions.

   The target may provide the needed syscalls by any of the following:

   1) Define the reentrant versions of the syscalls directly.
      (eg: _open_r, _close_r, etc.).  Please keep the namespace clean.
      When you do this, set "syscall_dir" to "syscalls" and add
      -DREENTRANT_SYSCALLS_PROVIDED to newlib_cflags in configure.host.

   2) Define namespace clean versions of the system calls by prefixing
      them with '_' (eg: _open, _close, etc.).  Technically, there won't be
      true reentrancy at the syscall level, but the library will be namespace
      clean.
      When you do this, set "syscall_dir" to "syscalls" in configure.host.

   3) Define or otherwise provide the regular versions of the syscalls
      (eg: open, close, etc.).  The library won't be reentrant nor namespace
      clean, but at least it will work.
      When you do this, add -DMISSING_SYSCALL_NAMES to newlib_cflags in
      configure.host.

   4) Define or otherwise provide the regular versions of the syscalls,
      and do not supply functional interfaces for any of the reentrant
      calls. With this method, the reentrant syscalls are redefined to
      directly call the regular system call without the reentrancy argument.
      When you do this, specify both -DREENTRANT_SYSCALLS_PROVIDED and 
      -DMISSING_SYSCALL_NAMES via newlib_cflags in configure.host and do
      not specify "syscall_dir".

   Stubs of the reentrant versions of the syscalls exist in the libc/reent
   source directory and are provided if REENTRANT_SYSCALLS_PROVIDED isn't 
   defined.  These stubs call the native system calls: _open, _close, etc. 
   if MISSING_SYSCALL_NAMES is *not* defined, otherwise they call the
   non-underscored versions: open, close, etc. when MISSING_SYSCALL_NAMES
   *is* defined.

   By default, newlib functions call the reentrant syscalls internally,
   passing a reentrancy structure as an argument.  This reentrancy structure
   contains data that is thread-specific.  For example, the errno value is
   kept in the reentrancy structure.  If multiple threads exist, each will
   keep a separate errno value which is intuitive since the application flow
   cannot check for failure reliably otherwise.

   The reentrant syscalls are either provided by the platform, by the 
   libc/reent stubs, or in the case of both MISSING_SYSCALL_NAMES and 
   REENTRANT_SYSCALLS_PROVIDED being defined, the calls are redefined to
   simply call the regular syscalls with no reentrancy struct argument.

   A single-threaded application does not need to worry about the reentrancy
   structure.  It is used internally.  

   A multi-threaded application needs either to manually manage reentrancy 
   structures or use dynamic reentrancy.

   Dynamic reentrancy is specified by the __DYNAMIC_REENT__ flag.  This
   flag denotes setting up a macro to replace _REENT with a function call
   to __getreent().  This function needs to be implemented by the platform
   and it is meant to return the reentrancy structure for the current
   thread.  When the regular C functions (e.g. printf) go to call internal
   routines with the default _REENT structure, they end up calling with 
   the reentrancy structure for the thread.  Thus, application code does not
   need to call the _r routines nor worry about reentrancy structures.  */

/* WARNING: All identifiers here must begin with an underscore.  This file is
   included by stdio.h and others and we therefore must only use identifiers
   in the namespace allotted to us.  */

#ifndef _REENT_H_
#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif
#define _REENT_H_

#include <sys/reent.h>
#include <sys/_types.h>
#include <machine/types.h>

#define __need_size_t
#define __need_ptrdiff_t
#include <stddef.h>

/* FIXME: not namespace clean */
struct stat;
struct tms;
struct timeval;
struct timezone;

#if defined(REENTRANT_SYSCALLS_PROVIDED) && defined(MISSING_SYSCALL_NAMES)

#if defined(__xcore__)

extern int _close _PARAMS ((int));
extern _off_t _lseek _PARAMS ((int, _off_t, int));
extern int _open _PARAMS ((const char *, int, unsigned));
extern _ssize_t _read _PARAMS ((int, char *, size_t));
extern _ssize_t _write _PARAMS ((int, const char *, size_t));
int _system _PARAMS ((_CONST char *s));
int _remove _PARAMS ((_CONST char *filename));

#define _close_r(__reent, __fd)                   _close(__fd)
#define _execve_r(__reent, __f, __arg, __env)     _execve(__f, __arg, __env)
#define _fcntl_r(__reent, __fd, __cmd, __arg)     _fcntl(__fd, __cmd, __arg)
#define _fork_r(__reent)                          _fork()
#define _fstat_r(__reent, __fdes, __stat)         _fstat(__fdes, __stat)
/* always return 0 */
#define _getpid_r(__reent)                        0
/* all streams are considered to be interactive. */
#define _isatty_r(__reent, __desc)                1
#define _kill_r(__reent, __pid, __signal)         _kill(__pid, __signal)
#define _link_r(__reent, __oldpath, __newpath)    _link(__oldpath, __newpath)
#define _lseek_r(__reent, __fdes, __off, __w)     _lseek(__fdes, __off, __w)
#define _open_r(__reent, __path, __flag, __m)     _open(__path, __flag, __m)
#define _read_r(__reent, __fd, __buff, __cnt)     _read(__fd, __buff, __cnt)
#define _sbrk_r(__reent, __incr)                  _sbrk(__incr)
#define _stat_r(__reent, __path, __buff)          _stat(__path, __buff)
#define _times_r(__reent, __time)                 _times(__time)
/* unlink is called _remove. */
#define _unlink_r(__reent, __path)                _remove(__path)
#define _wait_r(__reent, __status)                _wait(__status)
#define _write_r(__reent, __fd, __buff, __cnt)    _write(__fd, __buff, __cnt)
#define _gettimeofday_r(__reent, __tp, __tzp)     _gettimeofday(__tp, __tzp)

#ifdef HAVE_RENAME
#define _rename_r(__reent, __old, __new)          _rename(__old, __new)
#endif /* HAVE_RENAME */

#else

#define _close_r(__reent, __fd)                   close(__fd)
#define _execve_r(__reent, __f, __arg, __env)     execve(__f, __arg, __env)
#define _fcntl_r(__reent, __fd, __cmd, __arg)     fcntl(__fd, __cmd, __arg)
#define _fork_r(__reent)                          fork()
#define _fstat_r(__reent, __fdes, __stat)         fstat(__fdes, __stat)
#define _getpid_r(__reent)                        getpid()
#define _isatty_r(__reent, __desc)                isatty(__desc)
#define _kill_r(__reent, __pid, __signal)         kill(__pid, __signal)
#define _link_r(__reent, __oldpath, __newpath)    link(__oldpath, __newpath)
#define _lseek_r(__reent, __fdes, __off, __w)     lseek(__fdes, __off, __w)
#define _open_r(__reent, __path, __flag, __m)     open(__path, __flag, __m)
#define _read_r(__reent, __fd, __buff, __cnt)     read(__fd, __buff, __cnt)
#define _sbrk_r(__reent, __incr)                  sbrk(__incr)
#define _stat_r(__reent, __path, __buff)          stat(__path, __buff)
#define _times_r(__reent, __time)                 times(__time)
#define _unlink_r(__reent, __path)                unlink(__path)
#define _wait_r(__reent, __status)                wait(__status)
#define _write_r(__reent, __fd, __buff, __cnt)    write(__fd, __buff, __cnt)
#define _gettimeofday_r(__reent, __tp, __tzp)     gettimeofday(__tp, __tzp)

#endif /* defined(__xcore__) */

#ifdef __LARGE64_FILES
#define _lseek64_r(__reent, __fd, __off, __w)     lseek64(__fd, __off, __w)
#define _fstat64_r(__reent, __fd, __buff)         fstat64(__fd, __buff)
#define _open64_r(__reent, __path, __flag, __m)   open64(__path, __flag, __m)
#endif

#else
/* Reentrant versions of system calls.  */

extern int _close_r _PARAMS ((struct _reent *, int));
extern int _execve_r _PARAMS ((struct _reent *, char *, char **, char **));
extern int _fcntl_r _PARAMS ((struct _reent *, int, int, int));
extern int _fork_r _PARAMS ((struct _reent *));
extern int _fstat_r _PARAMS ((struct _reent *, int, struct stat *));
extern int _getpid_r _PARAMS ((struct _reent *));
extern int _isatty_r _PARAMS ((struct _reent *, int));
extern int _kill_r _PARAMS ((struct _reent *, int, int));
extern int _link_r _PARAMS ((struct _reent *, const char *, const char *));
extern _off_t _lseek_r _PARAMS ((struct _reent *, int, _off_t, int));
extern int _open_r _PARAMS ((struct _reent *, const char *, int, int));
extern _ssize_t _read_r _PARAMS ((struct _reent *, int, void *, size_t));
extern void *_sbrk_r _PARAMS ((struct _reent *, ptrdiff_t));
extern int _stat_r _PARAMS ((struct _reent *, const char *, struct stat *));
extern _CLOCK_T_ _times_r _PARAMS ((struct _reent *, struct tms *));
extern int _unlink_r _PARAMS ((struct _reent *, const char *));
extern int _wait_r _PARAMS ((struct _reent *, int *));
extern _ssize_t _write_r _PARAMS ((struct _reent *, int, const void *, size_t));

/* This one is not guaranteed to be available on all targets.  */
extern int _gettimeofday_r _PARAMS ((struct _reent *, struct timeval *__tp, void *__tzp));

#ifdef __LARGE64_FILES

#if defined(__CYGWIN__) && defined(_COMPILING_NEWLIB)
#define stat64 __stat64
#endif

struct stat64;

extern _off64_t _lseek64_r _PARAMS ((struct _reent *, int, _off64_t, int));
extern int _fstat64_r _PARAMS ((struct _reent *, int, struct stat64 *));
extern int _open64_r _PARAMS ((struct _reent *, const char *, int, int));
#endif

#endif

#if defined(__cplusplus) || defined(__XC__)
}
#endif
#endif /* _REENT_H_ */
