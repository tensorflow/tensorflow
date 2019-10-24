#ifndef _xc_errno_h_
#define _xc_errno_h_

#include_next <errno.h>
#include <safe/errno.h>

#if defined(__XC__)
#define _safe_errno (*_safe_errno_addr())
#if defined(UNSAFE_LIBC)
#define errno (*__errno())
#else
#define errno _safe_errno
#endif /* defined(UNSAFE_LIBC) */
#endif /* defined(__XC__) */

#endif /* _xc_errno_h_ */
