#ifndef _SYS_WAIT_H
#define _SYS_WAIT_H

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

#include <sys/types.h>

#define WNOHANG 1
#define WUNTRACED 2

/* A status looks like:
      <2 bytes info> <2 bytes code>

      <code> == 0, child has exited, info is the exit value
      <code> == 1..7e, child has exited, info is the signal number.
      <code> == 7f, child has stopped, info was the signal number.
      <code> == 80, there was a core dump.
*/
   
#define WIFEXITED(w)	(((w) & 0xff) == 0)
#define WIFSIGNALED(w)	(((w) & 0x7f) > 0 && (((w) & 0x7f) < 0x7f))
#define WIFSTOPPED(w)	(((w) & 0xff) == 0x7f)
#define WEXITSTATUS(w)	(((w) >> 8) & 0xff)
#define WTERMSIG(w)	((w) & 0x7f)
#define WSTOPSIG	WEXITSTATUS

pid_t wait (int *);
pid_t waitpid (pid_t, int *, int);

/* Provide prototypes for most of the _<systemcall> names that are
   provided in newlib for some compilers.  */
pid_t _wait (int *);

#if defined(__cplusplus) || defined(__XC__)
};
#endif

#endif
