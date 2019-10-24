/* <dirent.h> includes <sys/dirent.h>, which is this file.  On a
   system which supports <dirent.h>, this file is overridden by
   dirent.h in the libc/sys/.../sys directory.  On a system which does
   not support <dirent.h>, we will get this file which uses #error to force
   an error.  */

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif
#error "<dirent.h> not supported"
#if defined(__cplusplus) || defined(__XC__)
}
#endif
