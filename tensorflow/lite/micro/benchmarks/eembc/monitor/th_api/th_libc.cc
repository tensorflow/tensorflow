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

/**
 * These functions are needed by the main framework. If no LIBC
 * is provided, please implmenent these.
 */

#include "tensorflow/lite/micro/benchmarks/eembc/monitor/th_api/th_libc.h"

int
th_strncmp(const char *str1, const char *str2, size_t n)
{
    return strncmp(str1, str2, n);
}

size_t
th_strlen(const char * str)
{
    return strlen(str);
}

/*@-mayaliasunique*/
/*@-temptrans*/
char *
th_strcat(char *dest, const char *src)
{
    return strcat(dest, src);
}

char *
th_strstr(const char *str1, const char *str2)
{
    return strstr(str1, str2);
}

char *
th_strtok(char *str1, const char *sep)
{
    return strtok(str1, sep);
}

int
th_atoi(const char *str)
{
    return atoi(str);
}

void *
th_memset(void *b, int c, size_t len)
{
    return memset(b, c, len);
}

void *
th_memcpy(void *dst, const void *src, size_t n)
{
    return memcpy(dst, src, n);
}

void *
th_malloc(size_t size)
{
    return malloc(size);
}

void *
th_calloc(size_t count, size_t size)
{
    return calloc(count, size);
}

void
th_free(void *ptr)
{
    free(ptr);
}

int
th_vprintf(const char *format, va_list ap)
{
    return vprintf(format, ap);
}
