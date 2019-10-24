/*
 * Copyright (C) XMOS Limited 2008
 * 
 * The copyrights, all other intellectual and industrial property rights are
 * retained by XMOS and/or its licensors.
 *
 * The code is provided "AS IS" without a warranty of any kind. XMOS and its
 * licensors disclaim all other warranties, express or implied, including any
 * implied warranty of merchantability/satisfactory quality, fitness for a
 * particular purpose, or non-infringement except to the extent that these
 * disclaimers are held to be legally invalid under applicable law.
 *
 * Version: Community_15.0.0_eng
 */

#ifndef _SYSCALL_H_
#define _SYSCALL_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Standard streams */
#define FD_STDIN  0
#define FD_STDOUT 1
#define FD_STDERR 2

/* flag for lseek */
#define SEEK_CUR        1
#define SEEK_END        2
#define SEEK_SET        0

/* modes for open */
#define S_IREAD         0400            /* read permission */
#define S_IWRITE        0200            /* write permission */
#define S_IEXEC         0100            /* execute permission */

/* flags for open */
#define O_RDONLY    0x0001              /* read only */
#define O_WRONLY    0x0002              /* write only */
#define O_RDWR      0x0004              /* read/write enable */
#define O_CREAT     0x0100              /* create and open file */
#define O_TRUNC     0x0200              /* open with truncation */
#define	O_EXCL      0x0400              /* exclusive open */
#define O_APPEND    0x0800              /* to end of file */
#define O_BINARY    0x8000              /* no translation */

typedef unsigned ___size_t;

typedef unsigned ___mode_t;

typedef long ___off_t;

typedef long ___time_t;

void _exit(int status);
void _done();
int _open(const char path[], int flags, ___mode_t mode);
int _close(int d);
int _read(int fd, char buf[], unsigned count);
int _write(int fd, const char buf[], ___size_t count);
___off_t _lseek(int fd, ___off_t offset, int origin);
int _remove(const char filename[]);
int _rename(const char oldname[], const char newname[]);
#if defined(__XC__)
int _system(const char (&?s)[]);
#else
int _system(const char *);
#endif
#if defined(__STDC__) || !defined(__XC__)
___time_t _time(___time_t *t);
#else
___time_t _time(___time_t &?t);
#endif
void _exception(unsigned type, unsigned data);
int _is_simulation(void);

#if defined(__XC__)
int _load_image(char dst[count], unsigned int src, ___size_t count);
#else
int _load_image(void *dst, unsigned int src, ___size_t count);
#endif

#if defined(__XC__)
extern "C" {
#endif
int _get_cmdline(void *buf, unsigned size);
#if defined(__XC__)
} //extern "C"
#endif

/*
 * Plugins
 */
#define NOTIFY_PLUGINS_START_TRACE 0
#define NOTIFY_PLUGINS_STOP_TRACE  1

void _plugins(int type, unsigned arg1, unsigned arg2);

#define _traceStart() _plugins(NOTIFY_PLUGINS_START_TRACE, 0, 0)
#define _traceStop()  _plugins(NOTIFY_PLUGINS_STOP_TRACE, 0, 0)

#ifdef __cplusplus
} //extern "C" 
#endif

#endif /* _SYSCALL_H_ */
