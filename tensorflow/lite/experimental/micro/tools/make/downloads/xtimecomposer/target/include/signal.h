#ifndef _SIGNAL_H_
#define _SIGNAL_H_

#include "_ansi.h"
#include <sys/signal.h>

_BEGIN_STD_C

typedef int	sig_atomic_t;		/* Atomic entity type (ANSI) */

#ifndef __XC__
#define SIG_DFL ((_sig_func_ptr)0)	/* Default action */
#define SIG_IGN ((_sig_func_ptr)1)	/* Ignore action */
#define SIG_ERR ((_sig_func_ptr)-1)	/* Error return */

// xcore requires arg '_sig_func_ptr' to have its fptrgroup attribute set viz:
//    __attribute__((fptrgroup("signal_sig_func"))) void mySigHandler(int) {...}
_sig_func_ptr _EXFUN(signal, (int, _sig_func_ptr));
#endif /* __XC__ */
int	_EXFUN(raise, (int));

_END_STD_C

#endif /* _SIGNAL_H_ */
