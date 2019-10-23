#ifndef _DIRENT_H_
#define _DIRENT_H_
#if  defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif
#include <sys/dirent.h>

#if !defined(MAXNAMLEN) && !defined(_POSIX_SOURCE)
#define MAXNAMLEN 1024
#endif

#if  defined(__cplusplus) || defined(__XC__)
}
#endif
#endif /*_DIRENT_H_*/
