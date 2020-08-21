// Copyright (c) 2019, XMOS Ltd, All rights reserved

#include "fifo.h"

#include <string.h>

size_t fifo_level(fifo_t fifo)
{
    return fifo->total_written - fifo->total_read;
}

int fifo_full(fifo_t fifo)
{
    return fifo_level(fifo) >= fifo->buffer_length;
}

int fifo_empty(fifo_t fifo)
{
    return fifo_level(fifo) == 0;
}

int fifo_ready(fifo_t fifo)
{
    return fifo->ready;
}

int fifo_put(fifo_t fifo, void *element)
{
    unsigned tmp_head = fifo->head;

    if (!fifo_full(fifo)) {

        memcpy(fifo->buffer + tmp_head, element, fifo->element_size);

        tmp_head += fifo->element_size;

        if (tmp_head >= fifo->buffer_length) {
            fifo->head = 0;
        } else {
            fifo->head = tmp_head;
        }

        fifo->total_written += fifo->element_size;

        if (!fifo->ready && fifo_level(fifo) >= fifo->ready_level) {
            fifo->ready = 1;
        }

        return 0;
    } else {
        return -1;
    }
}

void fifo_put_blocking(fifo_t fifo, void *element)
{
    while (fifo_full(fifo));

    (void) fifo_put(fifo, element);
}

int fifo_get(fifo_t fifo, void *element)
{
    unsigned tmp_tail = fifo->tail;

    if (fifo_ready(fifo)) {
        memcpy(element, fifo->buffer + tmp_tail, fifo->element_size);

        tmp_tail += fifo->element_size;

        if (tmp_tail >= fifo->buffer_length) {
            fifo->tail = 0;
        } else {
            fifo->tail = tmp_tail;
        }

        fifo->total_read += fifo->element_size;

        if (fifo_level(fifo) == 0) {
            fifo->ready = 0;
        }

        return 0;
    } else {
        memset(element, 0, fifo->element_size);

        return -1;
    }
}

void fifo_get_blocking(fifo_t fifo, void *element)
{
    while (!fifo_ready(fifo));

    (void) fifo_get(fifo, element);
}
