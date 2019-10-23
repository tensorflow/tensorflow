/*
 * libgen.h - defined by XPG4
 */

#ifndef _LIBGEN_H_
#define _LIBGEN_H_

#include "_ansi.h"
#include <sys/reent.h>

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

char      *_EXFUN(basename,     (char *));
char      *_EXFUN(dirname,     (char *));

#if defined(__cplusplus) || defined(__XC__)
}
#endif

#endif /* _LIBGEN_H_ */

