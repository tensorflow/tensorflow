#ifndef _xc_stdio_h_
#define _xc_stdio_h_

#include_next <stdio.h>
#include <safe/stdio.h>

#if defined(__XC__) && !defined(UNSAFE_LIBC)
#define tmpfile() _safe_tmpfile()
#define tmpnam(s) _safe_tmpnam(s)
#define freopen(path, mode, fp) _safe_freopen(path, mode, fp)
#define fgets(s, size, fp) _safe_fgets(s, size, fp)
#define fputs(s, fp) _safe_fputs(s, fp)
#define gets(s) _safe_gets(s)
#define puts(s) _safe_puts(s)
#define fread(ptr, size, n, fp) _safe_fread((char *)ptr, size, n, fp)
#define fwrite(ptr, size, n, fp) _safe_fwrite((const char *)ptr, size, n, fp)
#define fgetpos(fp, pos) _safe_fgetpos(fp, pos)
#define fsetpos(fp, pos) _safe_fsetpos(fp, pos)
#define perror(s) _safe_perror(s)
#define fopen(name, type) _safe_fopen(name, type)
#define fclose(file) _safe_fclose(file)
#define remove(file) _safe_remove(file)
#define rename(from, to) _safe_rename(from, to)
#endif

#endif /* _xc_stdio_h_ */
