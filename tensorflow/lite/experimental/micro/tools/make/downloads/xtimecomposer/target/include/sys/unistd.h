#ifndef _SYS_UNISTD_H
#define _SYS_UNISTD_H

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

#include <_ansi.h>
#include <sys/types.h>
#include <sys/_types.h>
#define __need_size_t
#define __need_ptrdiff_t
#include <stddef.h>

extern char **environ;

void	_EXFUN(_exit, (int __status ) _ATTRIBUTE ((noreturn)));

int	_EXFUN(access,(const char *__path, int __amode ));
unsigned  _EXFUN(alarm, (unsigned __secs ));
int     _EXFUN(chdir, (const char *__path ));
int     _EXFUN(chmod, (const char *__path, mode_t __mode ));
#if !defined(__INSIDE_CYGWIN__)
int     _EXFUN(chown, (const char *__path, uid_t __owner, gid_t __group ));
#endif
#if defined(__CYGWIN__) || defined(__rtems__)
int     _EXFUN(chroot, (const char *__path ));
#endif
int     _EXFUN(close, (int __fildes ));
#if defined(__CYGWIN__)
size_t	_EXFUN(confstr, (int __name, char *__buf, size_t __len));
#endif
char    _EXFUN(*ctermid, (char *__s ));
char    _EXFUN(*cuserid, (char *__s ));
#if defined(__CYGWIN__)
int	_EXFUN(daemon, (int nochdir, int noclose));
#endif
int     _EXFUN(dup, (int __fildes ));
int     _EXFUN(dup2, (int __fildes, int __fildes2 ));
#if defined(__CYGWIN__)
void	_EXFUN(endusershell, (void));
#endif
int     _EXFUN(execl, (const char *__path, const char *, ... ));
int     _EXFUN(execle, (const char *__path, const char *, ... ));
int     _EXFUN(execlp, (const char *__file, const char *, ... ));
int     _EXFUN(execv, (const char *__path, char * const *__argv ));
int     _EXFUN(execve, (const char *__path, char * const *__argv, char * const *__envp ));
int     _EXFUN(execvp, (const char *__file, char * const *__argv ));
#if defined(__CYGWIN__)
int	_EXFUN(faccessat, (int __dirfd, const char *__path, int __mode, int __flags));
#endif
#if defined(__CYGWIN__) || defined(__rtems__) || defined(__SPU__)
int     _EXFUN(fchdir, (int __fildes));
#endif
int     _EXFUN(fchmod, (int __fildes, mode_t __mode ));
#if !defined(__INSIDE_CYGWIN__)
int     _EXFUN(fchown, (int __fildes, uid_t __owner, gid_t __group ));
#endif
#if defined(__CYGWIN__)
int	_EXFUN(fchownat, (int __dirfd, const char *__path, uid_t __owner, gid_t __group, int __flags));
#endif
pid_t   _EXFUN(fork, (void ));
long    _EXFUN(fpathconf, (int __fd, int __name ));
int     _EXFUN(fsync, (int __fd));
int     _EXFUN(fdatasync, (int __fd));
char    _EXFUN(*getcwd, (char *__buf, size_t __size ));
#if defined(__CYGWIN__)
int	_EXFUN(getdomainname ,(char *__name, size_t __len));
#endif
#if !defined(__INSIDE_CYGWIN__)
gid_t   _EXFUN(getegid, (void ));
uid_t   _EXFUN(geteuid, (void ));
gid_t   _EXFUN(getgid, (void ));
#endif
int     _EXFUN(getgroups, (int __gidsetsize, gid_t *__grouplist ));
#if defined(__CYGWIN__)
long    _EXFUN(gethostid, (void));
#endif
char    _EXFUN(*getlogin, (void ));
#if defined(_POSIX_THREAD_SAFE_FUNCTIONS)
int _EXFUN(getlogin_r, (char *name, size_t namesize) );
#endif
char 	_EXFUN(*getpass, (const char *__prompt));
size_t  _EXFUN(getpagesize, (void));
#if defined(__CYGWIN__)
int    _EXFUN(getpeereid, (int, uid_t *, gid_t *));
#endif
pid_t   _EXFUN(getpgid, (pid_t));
pid_t   _EXFUN(getpgrp, (void ));
pid_t   _EXFUN(getpid, (void ));
pid_t   _EXFUN(getppid, (void ));
#ifdef __CYGWIN__
pid_t   _EXFUN(getsid, (pid_t));
#endif
#if !defined(__INSIDE_CYGWIN__)
uid_t   _EXFUN(getuid, (void ));
#endif
#ifdef __CYGWIN__
char *	_EXFUN(getusershell, (void));
char    _EXFUN(*getwd, (char *__buf ));
int	_EXFUN(iruserok, (unsigned long raddr, int superuser, const char *ruser, const char *luser));
#endif
int     _EXFUN(isatty, (int __fildes ));
#if !defined(__INSIDE_CYGWIN__)
int     _EXFUN(lchown, (const char *__path, uid_t __owner, gid_t __group ));
#endif
int     _EXFUN(link, (const char *__path1, const char *__path2 ));
#if defined(__CYGWIN__)
int     _EXFUN(linkat, (int __dirfd1, const char *__path1, int __dirfd2, const char *__path2, int __flags ));
#endif
int	_EXFUN(nice, (int __nice_value ));
#if !defined(__INSIDE_CYGWIN__)
off_t   _EXFUN(lseek, (int __fildes, off_t __offset, int __whence ));
#endif
#if defined(__SPU__) || defined(__CYGWIN__)
#define F_ULOCK	0
#define F_LOCK	1
#define F_TLOCK	2
#define F_TEST	3
int     _EXFUN(lockf, (int __fd, int __cmd, off_t __len));
#endif
long    _EXFUN(pathconf, (const char *__path, int __name ));
int     _EXFUN(pause, (void ));
#ifdef __CYGWIN__
int	_EXFUN(pthread_atfork, (void (*)(void), void (*)(void), void (*)(void)));
#endif
int     _EXFUN(pipe, (int __fildes[2] ));
ssize_t _EXFUN(pread, (int __fd, void *__buf, size_t __nbytes, off_t __offset));
ssize_t _EXFUN(pwrite, (int __fd, const void *__buf, size_t __nbytes, off_t __offset));
_READ_WRITE_RETURN_TYPE _EXFUN(read, (int __fd, void *__buf, size_t __nbyte ));
#if defined(__CYGWIN__)
int	_EXFUN(rresvport, (int *__alport));
int	_EXFUN(revoke, (char *__path));
#endif
int     _EXFUN(rmdir, (const char *__path ));
#if defined(__CYGWIN__)
int	_EXFUN(ruserok, (const char *rhost, int superuser, const char *ruser, const char *luser));
#endif
void *  _EXFUN(sbrk,  (ptrdiff_t __incr));
#if !defined(__INSIDE_CYGWIN__)
#if defined(__CYGWIN__)
int     _EXFUN(setegid, (gid_t __gid ));
int     _EXFUN(seteuid, (uid_t __uid ));
#endif
int     _EXFUN(setgid, (gid_t __gid ));
#endif
#if defined(__CYGWIN__)
int	_EXFUN(setgroups, (int ngroups, const gid_t *grouplist ));
#endif
int     _EXFUN(setpgid, (pid_t __pid, pid_t __pgid ));
int     _EXFUN(setpgrp, (void ));
#if defined(__CYGWIN__) && !defined(__INSIDE_CYGWIN__)
int	_EXFUN(setregid, (gid_t __rgid, gid_t __egid));
int	_EXFUN(setreuid, (uid_t __ruid, uid_t __euid));
#endif
pid_t   _EXFUN(setsid, (void ));
#if !defined(__INSIDE_CYGWIN__)
int     _EXFUN(setuid, (uid_t __uid ));
#endif
#if defined(__CYGWIN__)
void	_EXFUN(setusershell, (void));
#endif
unsigned _EXFUN(sleep, (unsigned int __seconds ));
void    _EXFUN(swab, (const void *, void *, ssize_t));
long    _EXFUN(sysconf, (int __name ));
pid_t   _EXFUN(tcgetpgrp, (int __fildes ));
int     _EXFUN(tcsetpgrp, (int __fildes, pid_t __pgrp_id ));
char    _EXFUN(*ttyname, (int __fildes ));
#if defined(__CYGWIN__)
int     _EXFUN(ttyname_r, (int, char *, size_t)); 
#endif
int     _EXFUN(unlink, (const char *__path ));
int 	_EXFUN(usleep, (useconds_t __useconds));
int     _EXFUN(vhangup, (void ));
_READ_WRITE_RETURN_TYPE _EXFUN(write, (int __fd, const void *__buf, size_t __nbyte ));

#ifdef __CYGWIN__
# define __UNISTD_GETOPT__
# include <getopt.h>
# undef __UNISTD_GETOPT__
#else
extern char *optarg;			/* getopt(3) external variables */
extern int optind, opterr, optopt;
int	 getopt(int, char * const *, const char *);
extern int optreset;			/* getopt(3) external variable */
#endif

#ifndef        _POSIX_SOURCE
pid_t   _EXFUN(vfork, (void ));

extern char *suboptarg;			/* getsubopt(3) external variable */
int	 getsubopt(char **, char * const *, char **);
#endif /* _POSIX_SOURCE */

#ifdef _COMPILING_NEWLIB
/* Provide prototypes for most of the _<systemcall> names that are
   provided in newlib for some compilers.  */
int     _EXFUN(_close, (int __fildes ));
pid_t   _EXFUN(_fork, (void ));
pid_t   _EXFUN(_getpid, (void ));
int	_EXFUN(_isatty, (int __fildes ));
int     _EXFUN(_link, (const char *__path1, const char *__path2 ));
_off_t   _EXFUN(_lseek, (int __fildes, _off_t __offset, int __whence ));
#ifdef __LARGE64_FILES
_off64_t _EXFUN(_lseek64, (int __filedes, _off64_t __offset, int __whence ));
#endif
_READ_WRITE_RETURN_TYPE _EXFUN(_read, (int __fd, void *__buf, size_t __nbyte ));
void *  _EXFUN(_sbrk,  (ptrdiff_t __incr));
int     _EXFUN(_unlink, (const char *__path ));
_READ_WRITE_RETURN_TYPE _EXFUN(_write, (int __fd, const void *__buf, size_t __nbyte ));
int     _EXFUN(_execve, (const char *__path, char * const *__argv, char * const *__envp ));
#endif

#if defined(__CYGWIN__) || defined(__rtems__) || defined(__sh__) || defined(__SPU__)
#if !defined(__INSIDE_CYGWIN__)
int     _EXFUN(ftruncate, (int __fd, off_t __length));
int     _EXFUN(truncate, (const char *, off_t __length));
#endif
#endif

#if defined(__CYGWIN__) || defined(__rtems__)
int	_EXFUN(getdtablesize, (void));
int	_EXFUN(setdtablesize, (int));
useconds_t _EXFUN(ualarm, (useconds_t __useconds, useconds_t __interval));
#if !(defined  (_WINSOCK_H) || defined (__USE_W32_SOCKETS))
/* winsock[2].h defines as __stdcall, and with int as 2nd arg */
 int	_EXFUN(gethostname, (char *__name, size_t __len));
#endif
char *	_EXFUN(mktemp, (char *));
#endif

#if defined(__CYGWIN__) || defined(__SPU__)
void    _EXFUN(sync, (void));
#elif defined(__rtems__)
int     _EXFUN(sync, (void));
#endif

ssize_t _EXFUN(readlink, (const char *__path, char *__buf, size_t __buflen));
#if defined(__CYGWIN__)
ssize_t	_EXFUN(readlinkat, (int __dirfd1, const char *__path, char *__buf, size_t __buflen));
#endif
int     _EXFUN(symlink, (const char *__name1, const char *__name2));

#define	F_OK	0
#define	R_OK	4
#define	W_OK	2
#define	X_OK	1

# define	SEEK_SET	0
# define	SEEK_CUR	1
# define	SEEK_END	2

#include <sys/features.h>

#define STDIN_FILENO    0       /* standard input file descriptor */
#define STDOUT_FILENO   1       /* standard output file descriptor */
#define STDERR_FILENO   2       /* standard error file descriptor */

/*
 *  sysconf values per IEEE Std 1003.1, 2004 Edition
 */

#define _SC_ARG_MAX                       0
#define _SC_CHILD_MAX                     1
#define _SC_CLK_TCK                       2
#define _SC_NGROUPS_MAX                   3
#define _SC_OPEN_MAX                      4
#define _SC_JOB_CONTROL                   5
#define _SC_SAVED_IDS                     6
#define _SC_VERSION                       7
#define _SC_PAGESIZE                      8
#define _SC_PAGE_SIZE                     _SC_PAGESIZE
/* These are non-POSIX values we accidentally introduced in 2000 without
   guarding them.  Keeping them unguarded for backward compatibility. */
#define _SC_NPROCESSORS_CONF              9
#define _SC_NPROCESSORS_ONLN             10
#define _SC_PHYS_PAGES                   11
#define _SC_AVPHYS_PAGES                 12
/* End of non-POSIX values. */
#define _SC_MQ_OPEN_MAX                  13
#define _SC_MQ_PRIO_MAX                  14
#define _SC_RTSIG_MAX                    15
#define _SC_SEM_NSEMS_MAX                16
#define _SC_SEM_VALUE_MAX                17
#define _SC_SIGQUEUE_MAX                 18
#define _SC_TIMER_MAX                    19
#define _SC_TZNAME_MAX                   20
#define _SC_ASYNCHRONOUS_IO              21
#define _SC_FSYNC                        22
#define _SC_MAPPED_FILES                 23
#define _SC_MEMLOCK                      24
#define _SC_MEMLOCK_RANGE                25
#define _SC_MEMORY_PROTECTION            26
#define _SC_MESSAGE_PASSING              27
#define _SC_PRIORITIZED_IO               28
#define _SC_REALTIME_SIGNALS             29
#define _SC_SEMAPHORES                   30
#define _SC_SHARED_MEMORY_OBJECTS        31
#define _SC_SYNCHRONIZED_IO              32
#define _SC_TIMERS                       33
#define _SC_AIO_LISTIO_MAX               34
#define _SC_AIO_MAX                      35
#define _SC_AIO_PRIO_DELTA_MAX           36
#define _SC_DELAYTIMER_MAX               37
#define _SC_THREAD_KEYS_MAX              38
#define _SC_THREAD_STACK_MIN             39
#define _SC_THREAD_THREADS_MAX           40
#define _SC_TTY_NAME_MAX                 41
#define _SC_THREADS                      42
#define _SC_THREAD_ATTR_STACKADDR        43
#define _SC_THREAD_ATTR_STACKSIZE        44
#define _SC_THREAD_PRIORITY_SCHEDULING   45
#define _SC_THREAD_PRIO_INHERIT          46
/* _SC_THREAD_PRIO_PROTECT was _SC_THREAD_PRIO_CEILING in early drafts */
#define _SC_THREAD_PRIO_PROTECT          47
#define _SC_THREAD_PRIO_CEILING          _SC_THREAD_PRIO_PROTECT
#define _SC_THREAD_PROCESS_SHARED        48
#define _SC_THREAD_SAFE_FUNCTIONS        49
#define _SC_GETGR_R_SIZE_MAX             50
#define _SC_GETPW_R_SIZE_MAX             51
#define _SC_LOGIN_NAME_MAX               52
#define _SC_THREAD_DESTRUCTOR_ITERATIONS 53
#define _SC_ADVISORY_INFO                54
#define _SC_ATEXIT_MAX                   55
#define _SC_BARRIERS                     56
#define _SC_BC_BASE_MAX                  57
#define _SC_BC_DIM_MAX                   58
#define _SC_BC_SCALE_MAX                 59
#define _SC_BC_STRING_MAX                60
#define _SC_CLOCK_SELECTION              61
#define _SC_COLL_WEIGHTS_MAX             62
#define _SC_CPUTIME                      63
#define _SC_EXPR_NEST_MAX                64
#define _SC_HOST_NAME_MAX                65
#define _SC_IOV_MAX                      66
#define _SC_IPV6                         67
#define _SC_LINE_MAX                     68
#define _SC_MONOTONIC_CLOCK              69
#define _SC_RAW_SOCKETS                  70
#define _SC_READER_WRITER_LOCKS          71
#define _SC_REGEXP                       72
#define _SC_RE_DUP_MAX                   73
#define _SC_SHELL                        74
#define _SC_SPAWN                        75
#define _SC_SPIN_LOCKS                   76
#define _SC_SPORADIC_SERVER              77
#define _SC_SS_REPL_MAX                  78
#define _SC_SYMLOOP_MAX                  79
#define _SC_THREAD_CPUTIME               80
#define _SC_THREAD_SPORADIC_SERVER       81
#define _SC_TIMEOUTS                     82
#define _SC_TRACE                        83
#define _SC_TRACE_EVENT_FILTER           84
#define _SC_TRACE_EVENT_NAME_MAX         85
#define _SC_TRACE_INHERIT                86
#define _SC_TRACE_LOG                    87
#define _SC_TRACE_NAME_MAX               88
#define _SC_TRACE_SYS_MAX                89
#define _SC_TRACE_USER_EVENT_MAX         90
#define _SC_TYPED_MEMORY_OBJECTS         91
#define _SC_V6_ILP32_OFF32               92
#define _SC_XBS5_ILP32_OFF32             _SC_V6_ILP32_OFF32
#define _SC_V6_ILP32_OFFBIG              93
#define _SC_XBS5_ILP32_OFFBIG            _SC_V6_ILP32_OFFBIG
#define _SC_V6_LP64_OFF64                94
#define _SC_XBS5_LP64_OFF64              _SC_V6_LP64_OFF64
#define _SC_V6_LPBIG_OFFBIG              95
#define _SC_XBS5_LPBIG_OFFBIG            _SC_V6_LPBIG_OFFBIG
#define _SC_XOPEN_CRYPT                  96
#define _SC_XOPEN_ENH_I18N               97
#define _SC_XOPEN_LEGACY                 98
#define _SC_XOPEN_REALTIME               99
#define _SC_STREAM_MAX                  100
#define _SC_PRIORITY_SCHEDULING         101
#define _SC_XOPEN_REALTIME_THREADS      102
#define _SC_XOPEN_SHM                   103
#define _SC_XOPEN_STREAMS               104
#define _SC_XOPEN_UNIX                  105
#define _SC_XOPEN_VERSION               106
#define _SC_2_CHAR_TERM                 107
#define _SC_2_C_BIND                    108
#define _SC_2_C_DEV                     109
#define _SC_2_FORT_DEV                  110
#define _SC_2_FORT_RUN                  111
#define _SC_2_LOCALEDEF                 112
#define _SC_2_PBS                       113
#define _SC_2_PBS_ACCOUNTING            114
#define _SC_2_PBS_CHECKPOINT            115
#define _SC_2_PBS_LOCATE                116
#define _SC_2_PBS_MESSAGE               117
#define _SC_2_PBS_TRACK                 118
#define _SC_2_SW_DEV                    119
#define _SC_2_UPE                       120
#define _SC_2_VERSION                   121

/*
 *  pathconf values per IEEE Std 1003.1, 2004 Edition
 */

#define _PC_LINK_MAX                      0
#define _PC_MAX_CANON                     1
#define _PC_MAX_INPUT                     2
#define _PC_NAME_MAX                      3
#define _PC_PATH_MAX                      4
#define _PC_PIPE_BUF                      5
#define _PC_CHOWN_RESTRICTED              6
#define _PC_NO_TRUNC                      7
#define _PC_VDISABLE                      8
#define _PC_ASYNC_IO                      9
#define _PC_PRIO_IO                      10
#define _PC_SYNC_IO                      11
#define _PC_FILESIZEBITS                 12
#define _PC_2_SYMLINKS                   13
#define _PC_SYMLINK_MAX                  14
#ifdef __CYGWIN__
/* Ask for POSIX permission bits support. */
#define _PC_POSIX_PERMISSIONS            90
/* Ask for full POSIX permission support including uid/gid settings. */
#define _PC_POSIX_SECURITY               91
#endif

/*
 *  confstr values per IEEE Std 1003.1, 2004 Edition
 */

#ifdef __CYGWIN__	/* Only defined on Cygwin for now. */
#define _CS_PATH                               0
#define _CS_POSIX_V6_ILP32_OFF32_CFLAGS        1
#define _CS_XBS5_ILP32_OFF32_CFLAGS           _CS_POSIX_V6_ILP32_OFF32_CFLAGS
#define _CS_POSIX_V6_ILP32_OFF32_LDFLAGS       2
#define _CS_XBS5_ILP32_OFF32_LDFLAGS          _CS_POSIX_V6_ILP32_OFF32_LDFLAGS
#define _CS_POSIX_V6_ILP32_OFF32_LIBS          3
#define _CS_XBS5_ILP32_OFF32_LIBS             _CS_POSIX_V6_ILP32_OFF32_LIBS
#define _CS_XBS5_ILP32_OFF32_LINTFLAGS         4
#define _CS_POSIX_V6_ILP32_OFFBIG_CFLAGS       5
#define _CS_XBS5_ILP32_OFFBIG_CFLAGS          _CS_POSIX_V6_ILP32_OFFBIG_CFLAGS
#define _CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS      6
#define _CS_XBS5_ILP32_OFFBIG_LDFLAGS         _CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS
#define _CS_POSIX_V6_ILP32_OFFBIG_LIBS         7
#define _CS_XBS5_ILP32_OFFBIG_LIBS            _CS_POSIX_V6_ILP32_OFFBIG_LIBS
#define _CS_XBS5_ILP32_OFFBIG_LINTFLAGS        8
#define _CS_POSIX_V6_LP64_OFF64_CFLAGS         9
#define _CS_XBS5_LP64_OFF64_CFLAGS            _CS_POSIX_V6_LP64_OFF64_CFLAGS
#define _CS_POSIX_V6_LP64_OFF64_LDFLAGS       10
#define _CS_XBS5_LP64_OFF64_LDFLAGS           _CS_POSIX_V6_LP64_OFF64_LDFLAGS
#define _CS_POSIX_V6_LP64_OFF64_LIBS          11
#define _CS_XBS5_LP64_OFF64_LIBS              _CS_POSIX_V6_LP64_OFF64_LIBS
#define _CS_XBS5_LP64_OFF64_LINTFLAGS         12
#define _CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS      13
#define _CS_XBS5_LPBIG_OFFBIG_CFLAGS          _CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS
#define _CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS     14
#define _CS_XBS5_LPBIG_OFFBIG_LDFLAGS         _CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS
#define _CS_POSIX_V6_LPBIG_OFFBIG_LIBS        15
#define _CS_XBS5_LPBIG_OFFBIG_LIBS            _CS_POSIX_V6_LPBIG_OFFBIG_LIBS
#define _CS_XBS5_LPBIG_OFFBIG_LINTFLAGS       16
#define _CS_POSIX_V6_WIDTH_RESTRICTED_ENVS    17
#endif

#ifndef __CYGWIN__
# define	MAXPATHLEN	1024
#endif

#if defined(__cplusplus) || defined(__XC__)
}
#endif
#endif /* _SYS_UNISTD_H */
