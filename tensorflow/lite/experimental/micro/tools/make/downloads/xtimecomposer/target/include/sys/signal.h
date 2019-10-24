/* sys/signal.h */

#ifndef _SYS_SIGNAL_H
#define _SYS_SIGNAL_H
#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

#include "_ansi.h"
#include <sys/features.h>

/* #ifndef __STRICT_ANSI__*/

#if defined(_POSIX_THREADS)
#include <sys/types.h>   /* for pthread data types */
#endif

typedef unsigned long sigset_t;

#if defined(__rtems__)

#if defined(_POSIX_REALTIME_SIGNALS)

/* sigev_notify values
   NOTE: P1003.1c/D10, p. 34 adds SIGEV_THREAD.  */

#define SIGEV_NONE   1  /* No asynchronous notification shall be delivered */
                        /*   when the event of interest occurs. */
#define SIGEV_SIGNAL 2  /* A queued signal, with an application defined */
                        /*  value, shall be delivered when the event of */
                        /*  interest occurs. */
#define SIGEV_THREAD 3  /* A notification function shall be called to */
                        /*   perform notification. */

/*  Signal Generation and Delivery, P1003.1b-1993, p. 63
    NOTE: P1003.1c/D10, p. 34 adds sigev_notify_function and
          sigev_notify_attributes to the sigevent structure.  */

union sigval {
  int    sival_int;    /* Integer signal value */
  void  *sival_ptr;    /* Pointer signal value */
};

struct sigevent {
  int              sigev_notify;               /* Notification type */
  int              sigev_signo;                /* Signal number */
  union sigval     sigev_value;                /* Signal value */

#if defined(_POSIX_THREADS)
  void           (*sigev_notify_function)( union sigval );
                                               /* Notification function */
  pthread_attr_t  *sigev_notify_attributes;    /* Notification Attributes */
#endif
};

/* Signal Actions, P1003.1b-1993, p. 64 */
/* si_code values, p. 66 */

#define SI_USER    1    /* Sent by a user. kill(), abort(), etc */
#define SI_QUEUE   2    /* Sent by sigqueue() */
#define SI_TIMER   3    /* Sent by expiration of a timer_settime() timer */
#define SI_ASYNCIO 4    /* Indicates completion of asycnhronous IO */
#define SI_MESGQ   5    /* Indicates arrival of a message at an empty queue */

typedef struct {
  int          si_signo;    /* Signal number */
  int          si_code;     /* Cause of the signal */
  union sigval si_value;    /* Signal value */
} siginfo_t;
#endif

/*  3.3.8 Synchronously Accept a Signal, P1003.1b-1993, p. 76 */

#define SA_NOCLDSTOP 1   /* Do not generate SIGCHLD when children stop */
#define SA_SIGINFO   2   /* Invoke the signal catching function with */
                         /*   three arguments instead of one. */

/* struct sigaction notes from POSIX:
 *
 *  (1) Routines stored in sa_handler should take a single int as
 *      their argument although the POSIX standard does not require this.
 *  (2) The fields sa_handler and sa_sigaction may overlap, and a conforming
 *      application should not use both simultaneously.
 */

typedef void (*_sig_func_ptr)();

struct sigaction {
  int         sa_flags;       /* Special flags to affect behavior of signal */
  sigset_t    sa_mask;        /* Additional set of signals to be blocked */
                              /*   during execution of signal-catching */
                              /*   function. */
  union {
    _sig_func_ptr _handler;  /* SIG_DFL, SIG_IGN, or pointer to a function */
#if defined(_POSIX_REALTIME_SIGNALS)
    void      (*_sigaction)( int, siginfo_t *, void * );
#endif
  } _signal_handlers;
};

#define sa_handler    _signal_handlers._handler
#if defined(_POSIX_REALTIME_SIGNALS)
#define sa_sigaction  _signal_handlers._sigaction
#endif

#elif defined(__CYGWIN__)
#include <cygwin/signal.h>
#else
#define SA_NOCLDSTOP 1  /* only value supported now for sa_flags */

#ifdef __XC__
struct sigaction;
#else
typedef void (*_sig_func_ptr)(int);

struct sigaction 
{
	_sig_func_ptr sa_handler;
	sigset_t sa_mask;
	int sa_flags;
};
#endif /* __XC__ */

#endif /* defined(__rtems__) */

#define SIG_SETMASK 0	/* set mask with sigprocmask() */
#define SIG_BLOCK 1	/* set of signals to block */
#define SIG_UNBLOCK 2	/* set of signals to, well, unblock */

/* These depend upon the type of sigset_t, which right now 
   is always a long.. They're in the POSIX namespace, but
   are not ANSI. */
#define sigaddset(what,sig) (*(what) |= (1<<(sig)), 0)
#define sigdelset(what,sig) (*(what) &= ~(1<<(sig)), 0)
#define sigemptyset(what)   (*(what) = 0, 0)
#define sigfillset(what)    (*(what) = ~(0), 0)
#define sigismember(what,sig) (((*(what)) & (1<<(sig))) != 0)

int _EXFUN(sigprocmask, (int how, const sigset_t *set, sigset_t *oset));

#if defined(_POSIX_THREADS)
int _EXFUN(pthread_sigmask, (int how, const sigset_t *set, sigset_t *oset));
#endif

/* protos for functions found in winsup sources for CYGWIN */
#if defined(__CYGWIN__) || defined(__rtems__)
#undef sigaddset
#undef sigdelset
#undef sigemptyset
#undef sigfillset
#undef sigismember
/* The first argument to kill should be pid_t.  Right now
   <sys/types.h> always defines pid_t to be int.  If that ever
   changes, then we will need to do something else, perhaps along the
   lines of <machine/types.h>.  */
int _EXFUN(kill, (int, int));
int _EXFUN(killpg, (pid_t, int));
int _EXFUN(sigaction, (int, const struct sigaction *, struct sigaction *));
int _EXFUN(sigaddset, (sigset_t *, const int));
int _EXFUN(sigdelset, (sigset_t *, const int));
int _EXFUN(sigismember, (const sigset_t *, int));
int _EXFUN(sigfillset, (sigset_t *));
int _EXFUN(sigemptyset, (sigset_t *));
int _EXFUN(sigpending, (sigset_t *));
int _EXFUN(sigsuspend, (const sigset_t *));
int _EXFUN(sigpause, (int));

#if defined(_POSIX_THREADS)
#ifdef __CYGWIN__
#  ifndef _CYGWIN_TYPES_H
#    error You need the winsup sources or a cygwin installation to compile the cygwin version of newlib.
#  endif
#endif
int _EXFUN(pthread_kill, (pthread_t thread, int sig));
#endif

#if defined(_POSIX_REALTIME_SIGNALS)

/*  3.3.8 Synchronously Accept a Signal, P1003.1b-1993, p. 76
    NOTE: P1003.1c/D10, p. 39 adds sigwait().  */

int _EXFUN(sigwaitinfo, (const sigset_t *set, siginfo_t *info));
int _EXFUN(sigtimedwait,
  (const sigset_t *set, siginfo_t *info, const struct timespec  *timeout)
);
int _EXFUN(sigwait, (const sigset_t *set, int *sig));

/*  3.3.9 Queue a Signal to a Process, P1003.1b-1993, p. 78 */
int _EXFUN(sigqueue, (pid_t pid, int signo, const union sigval value));

#endif /* defined(_POSIX_REALTIME_SIGNALS) */

#endif /* defined(__CYGWIN__) || defined(__rtems__) */

/* #endif __STRICT_ANSI__ */

#if defined(___AM29K__)
/* These all need to be defined for ANSI C, but I don't think they are
   meaningful.  */
#define SIGABRT 1
#define SIGFPE 1
#define SIGILL 1
#define SIGINT 1
#define SIGSEGV 1
#define SIGTERM 1
/* These need to be defined for POSIX, and some others do too.  */
#define SIGHUP 1
#define SIGQUIT 1
#define NSIG 2
#elif defined(__GO32__)
#define SIGINT  1
#define SIGKILL 2
#define SIGPIPE 3
#define SIGFPE  4
#define SIGHUP  5
#define SIGTERM 6
#define SIGSEGV 7
#define SIGTSTP 8
#define SIGQUIT 9
#define SIGTRAP 10
#define SIGILL  11
#define SIGEMT  12
#define SIGALRM 13
#define SIGBUS  14
#define SIGLOST 15
#define SIGSTOP 16
#define SIGABRT 17
#define SIGUSR1	18
#define SIGUSR2	19
#define NSIG    20
#elif defined(__xcore__)
/* Minimum set required by ANSI C. */
#define SIGABRT 1
#define SIGFPE 2
#define SIGILL 3
#define SIGINT 4
#define SIGSEGV 5
#define SIGTERM 6
#define NSIG    7
#elif !defined(SIGTRAP)
#define	SIGHUP	1	/* hangup */
#define	SIGINT	2	/* interrupt */
#define	SIGQUIT	3	/* quit */
#define	SIGILL	4	/* illegal instruction (not reset when caught) */
#define	SIGTRAP	5	/* trace trap (not reset when caught) */
#define	SIGIOT	6	/* IOT instruction */
#define	SIGABRT 6	/* used by abort, replace SIGIOT in the future */
#define	SIGEMT	7	/* EMT instruction */
#define	SIGFPE	8	/* floating point exception */
#define	SIGKILL	9	/* kill (cannot be caught or ignored) */
#define	SIGBUS	10	/* bus error */
#define	SIGSEGV	11	/* segmentation violation */
#define	SIGSYS	12	/* bad argument to system call */
#define	SIGPIPE	13	/* write on a pipe with no one to read it */
#define	SIGALRM	14	/* alarm clock */
#define	SIGTERM	15	/* software termination signal from kill */

#if defined(__rtems__)
#define	SIGURG	16	/* urgent condition on IO channel */
#define	SIGSTOP	17	/* sendable stop signal not from tty */
#define	SIGTSTP	18	/* stop signal from tty */
#define	SIGCONT	19	/* continue a stopped process */
#define	SIGCHLD	20	/* to parent on child stop or exit */
#define	SIGCLD	20	/* System V name for SIGCHLD */
#define	SIGTTIN	21	/* to readers pgrp upon background tty read */
#define	SIGTTOU	22	/* like TTIN for output if (tp->t_local&LTOSTOP) */
#define	SIGIO	23	/* input/output possible signal */
#define	SIGPOLL	SIGIO	/* System V name for SIGIO */
#define	SIGWINCH 24	/* window changed */
#define	SIGUSR1 25	/* user defined signal 1 */
#define	SIGUSR2 26	/* user defined signal 2 */

/* Real-Time Signals Range, P1003.1b-1993, p. 61
   NOTE: By P1003.1b-1993, this should be at least RTSIG_MAX
         (which is a minimum of 8) signals.
 */
#define SIGRTMIN 27
#define SIGRTMAX 31
#define __SIGFIRSTNOTRT SIGHUP
#define __SIGLASTNOTRT  SIGUSR2

#define NSIG	32      /* signal 0 implied */

#elif defined(__svr4__)
/* svr4 specifics. different signals above 15, and sigaction. */
#define	SIGUSR1	16
#define SIGUSR2	17
#define SIGCLD	18
#define	SIGPWR	19
#define SIGWINCH 20
#define	SIGPOLL	22	/* 20 for x.out binaries!!!! */
#define	SIGSTOP	23	/* sendable stop signal not from tty */
#define	SIGTSTP	24	/* stop signal from tty */
#define	SIGCONT	25	/* continue a stopped process */
#define	SIGTTIN	26	/* to readers pgrp upon background tty read */
#define	SIGTTOU	27	/* like TTIN for output if (tp->t_local&LTOSTOP) */
#define NSIG	28	
#else
#define	SIGURG	16	/* urgent condition on IO channel */
#define	SIGSTOP	17	/* sendable stop signal not from tty */
#define	SIGTSTP	18	/* stop signal from tty */
#define	SIGCONT	19	/* continue a stopped process */
#define	SIGCHLD	20	/* to parent on child stop or exit */
#define	SIGCLD	20	/* System V name for SIGCHLD */
#define	SIGTTIN	21	/* to readers pgrp upon background tty read */
#define	SIGTTOU	22	/* like TTIN for output if (tp->t_local&LTOSTOP) */
#define	SIGIO	23	/* input/output possible signal */
#define	SIGPOLL	SIGIO	/* System V name for SIGIO */
#define	SIGXCPU	24	/* exceeded CPU time limit */
#define	SIGXFSZ	25	/* exceeded file size limit */
#define	SIGVTALRM 26	/* virtual time alarm */
#define	SIGPROF	27	/* profiling time alarm */
#define	SIGWINCH 28	/* window changed */
#define	SIGLOST 29	/* resource lost (eg, record-lock lost) */
#define	SIGUSR1 30	/* user defined signal 1 */
#define	SIGUSR2 31	/* user defined signal 2 */
#define NSIG	32      /* signal 0 implied */
#endif
#endif

#if defined(__cplusplus) || defined(__XC__)
}
#endif

#ifndef _SIGNAL_H_
/* Some applications take advantage of the fact that <sys/signal.h>
 * and <signal.h> are equivalent in glibc.  Allow for that here.  */
#include <signal.h>
#endif
#endif /* _SYS_SIGNAL_H */
