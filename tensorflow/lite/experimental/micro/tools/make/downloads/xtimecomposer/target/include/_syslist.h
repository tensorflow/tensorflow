/* internal use only -- mapping of "system calls" for libraries that lose
   and only provide C names, so that we end up in violation of ANSI */
#ifndef __SYSLIST_H
#define __SYSLIST_H

#ifdef MISSING_SYSCALL_NAMES
#if !(defined(REENTRANT_SYSCALLS_PROVIDED) && defined(__xcore__))
#define _close close
#define _execve execve
#define _fcntl fcntl
#define _fork fork
#define _fstat fstat
#define _getpid getpid
#define _gettimeofday gettimeofday
#define _isatty isatty
#define _kill kill
#define _link link
#define _lseek lseek
#define _open open
#define _read read
#define _sbrk sbrk
#define _stat stat
#define _times times
#define _unlink unlink
#define _wait wait
#define _write write
#endif
#endif /* MISSING_SYSCALL_NAMES */

#if defined MISSING_SYSCALL_NAMES || !defined HAVE_OPENDIR
/* If the system call interface is missing opendir, readdir, and
   closedir, there is an implementation of these functions in
   libc/posix that is implemented using open, getdents, and close. 
   Note, these functions are currently not in the libc/syscalls
   directory.  */
#define _opendir opendir
#define _readdir readdir
#define _closedir closedir
#endif /* MISSING_SYSCALL_NAMES || !HAVE_OPENDIR */

#endif /* !__SYSLIST_H_ */
