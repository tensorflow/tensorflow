/*
 * Copyright (C) EEMBC(R). All Rights Reserved
 *
 * All EEMBC Benchmark Software are products of EEMBC and are provided under the
 * terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
 * are proprietary intellectual properties of EEMBC and its Members and is
 * protected under all applicable laws, including all applicable copyright laws.
 *
 * If you received this EEMBC Benchmark Software without having a currently
 * effective EEMBC Benchmark License Agreement, you must discontinue use.
 */

#ifndef __TH_LIBC_H
#define __TH_LIBC_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

int     th_strncmp(const char *str1, const char *str2, size_t n);
size_t  th_strlen(const char *str);
char*   th_strcat(char *dest, const char *src);
/*@null@*/
char*   th_strstr(const char *str1, const char *str2);
/*@null@*/
char*   th_strtok(/*@null@*/ char *str1, const char *sep);
int     th_atoi(const char *str);
void*   th_memset(void *b, int c, size_t len);
void*   th_memcpy(void* dst, const void* src, size_t n);
/*@null@*//*@out@*/
void*   th_malloc(size_t size);
/*@null@*//*@out@*/
void*   th_calloc(size_t count, size_t size);
void    th_free(void* ptr);
int     th_vprintf(const char *format, va_list ap);

#endif // __TH_LIBC_H
