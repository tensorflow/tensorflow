#ifndef _SYS_UTIME_H
#define _SYS_UTIME_H

/* This is a dummy <sys/utime.h> file, not customized for any
   particular system.  If there is a utime.h in libc/sys/SYSDIR/sys,
   it will override this one.  */

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

struct utimbuf 
{
  time_t actime;
  time_t modtime; 
};

#if defined(__cplusplus) || defined(__XC__)
};
#endif

#endif /* _SYS_UTIME_H */
