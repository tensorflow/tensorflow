/*
 * Copyright (C) 1991 DJ Delorie
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms is permitted
 * provided that the above copyright notice and following paragraph are
 * duplicated in all such forms.
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

/* Modified to use SETJMP_DJ_H rather than SETJMP_H to avoid
   conflicting with setjmp.h.  Ian Taylor, Cygnus support, April,
   1993.  */

#ifndef _SETJMP_DJ_H_
#define _SETJMP_DJ_H_

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

typedef struct {
  unsigned long eax;
  unsigned long ebx;
  unsigned long ecx;
  unsigned long edx;
  unsigned long esi;
  unsigned long edi;
  unsigned long ebp;
  unsigned long esp;
  unsigned long eip;
} jmp_buf[1];

extern int setjmp(jmp_buf);
extern void longjmp(jmp_buf, int);

#if defined(__cplusplus) || defined(__XC__)
}
#endif

#endif
