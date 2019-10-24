#ifndef _COMPLEX_H
#define _COMPLEX_H

#include "_ansi.h"

_BEGIN_STD_C

#define complex _Complex

#if defined(__GNUC__) && \
    (__GNUC__ >= 3)
# define _Complex_I	(__extension__ 1.0iF)

# define I _Complex_I
#endif /* __GNUC__ */

_END_STD_C

#endif /* _COMPLEX_H */
