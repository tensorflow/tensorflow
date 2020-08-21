// Copyright (c) 2019, XMOS Ltd, All rights reserved

#ifndef FIFO_H_
#define FIFO_H_

#include <sys/types.h>
#include <stdint.h>

#ifdef __XC__
extern "C" {
#endif //__XC__

#ifdef __XC__
/*
 * can't use the volatile keyword
 * in the xC definition of the struct
 * below. This is fine since only the
 * C code in fifo.c requires that the
 * members be volatile.
 */
#define volatile
#endif

struct fifo {
    uint8_t * const buffer;
    const size_t buffer_length;    // in bytes
    const size_t element_size;     // in bytes
    const size_t ready_level;      // in bytes
    volatile int ready;
    volatile size_t total_read;    // in bytes
    volatile size_t total_written; // in bytes
    unsigned head;
    unsigned tail;
};

typedef struct fifo * fifo_t;

#ifdef __XC__
#undef volatile
#endif

#define fifo_init(fptr, buf_len, el_sz, rdy_lvl) \
uint8_t fptr##_buf[(buf_len)*(el_sz)] = {0};     \
struct fifo fptr##_f = {                         \
    (fptr##_buf),                                \
    (buf_len)*(el_sz),                           \
    (el_sz),                                     \
    (rdy_lvl)*(el_sz),                           \
    0,                                           \
    0,                                           \
    0,                                           \
    0,                                           \
    0};                                          \
fptr = &fptr##_f;


size_t fifo_level(fifo_t fifo);
int fifo_full(fifo_t fifo);
int fifo_empty(fifo_t fifo);
int fifo_ready(fifo_t fifo);
int fifo_put(fifo_t fifo, void *element);
void fifo_put_blocking(fifo_t fifo, void *element);
int fifo_get(fifo_t fifo, void *element);
void fifo_get_blocking(fifo_t fifo, void *element);

#ifdef __XC__
}
#endif //__XC__

#endif /* FIFO_H_ */
