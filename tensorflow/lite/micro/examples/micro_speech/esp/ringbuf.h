/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_ESP_RINGBUF_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_ESP_RINGBUF_H_

#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RB_FAIL ESP_FAIL
#define RB_ABORT -1
#define RB_WRITER_FINISHED -2
#define RB_READER_UNBLOCK -3

typedef struct ringbuf {
  char *name;
  uint8_t *base; /**< Original pointer */
  /* XXX: these need to be volatile? */
  uint8_t *volatile readptr;  /**< Read pointer */
  uint8_t *volatile writeptr; /**< Write pointer */
  volatile ssize_t fill_cnt;  /**< Number of filled slots */
  ssize_t size;               /**< Buffer size */
  xSemaphoreHandle can_read;
  xSemaphoreHandle can_write;
  xSemaphoreHandle lock;
  int abort_read;
  int abort_write;
  int writer_finished;  // to prevent infinite blocking for buffer read
  int reader_unblock;
} ringbuf_t;

ringbuf_t *rb_init(const char *rb_name, uint32_t size);
void rb_abort_read(ringbuf_t *rb);
void rb_abort_write(ringbuf_t *rb);
void rb_abort(ringbuf_t *rb);
void rb_reset(ringbuf_t *rb);
/**
 * @brief Special function to reset the buffer while keeping rb_write aborted.
 *        This rb needs to be reset again before being useful.
 */
void rb_reset_and_abort_write(ringbuf_t *rb);
void rb_stat(ringbuf_t *rb);
ssize_t rb_filled(ringbuf_t *rb);
ssize_t rb_available(ringbuf_t *rb);
int rb_read(ringbuf_t *rb, uint8_t *buf, int len, uint32_t ticks_to_wait);
int rb_write(ringbuf_t *rb, const uint8_t *buf, int len,
             uint32_t ticks_to_wait);
void rb_cleanup(ringbuf_t *rb);
void rb_signal_writer_finished(ringbuf_t *rb);
void rb_wakeup_reader(ringbuf_t *rb);
int rb_is_writer_finished(ringbuf_t *rb);

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_ESP_RINGBUF_H_
