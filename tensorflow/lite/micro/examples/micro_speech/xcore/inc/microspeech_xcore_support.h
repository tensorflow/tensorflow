// Copyright (c) 2020, XMOS Ltd, All rights reserved

#ifndef MICROSPEECH_XCORE_SUPPORT_H_
#define MICROSPEECH_XCORE_SUPPORT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "fifo.h"

#if __XC__
typedef struct microspeech_device {
    fifo_t* unsafe sample_fifo;
    int32_t* unsafe sample_buffer;
} microspeech_device_t;
#else
typedef struct microspeech_device {
    fifo_t* sample_fifo;
    int32_t* sample_buffer;
} microspeech_device_t;
#endif

#if __XC__
microspeech_device_t* unsafe get_microspeech_device();
void mic_decoupler(streaming chanend c_ds_output, chanend c_gpio);
void tile0(chanend c_gpio);
#else
microspeech_device_t* get_microspeech_device();
void mic_decoupler( chanend_t c_ds_output, chanend_t c_gpio);
void tile0(chanend_t c_gpio);
#endif

void increment_timestamp(int32_t increment);
int32_t get_led_status();

#ifdef __cplusplus
}
#endif

#endif /* MICROSPEECH_XCORE_SUPPORT_H_ */
