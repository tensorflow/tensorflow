// Copyright (c) 2020, XMOS Ltd, All rights reserved

#include "tensorflow/lite/micro/examples/micro_speech/xcore/main_support.h"

#include <platform.h>

#include <xcore/triggerable.h>
#include <xcore/port.h>
#include <xcore/channel.h>

#include "mic_array_conf.h"
#include "fifo.h"
#include "microspeech_xcore_support.h"

const port_t p_gpio_leds = XS1_PORT_4C;

#define MIC_FRAME_BUFFER_COUNT  5
static int32_t mic_samples[MIC_FRAME_BUFFER_COUNT][(1 << MIC_ARRAY_MAX_FRAME_SIZE_LOG2)];
static microspeech_device_t mic_device;

static int32_t* mic_samples_ptr = NULL;
static fifo_t* fifo_ptr = NULL;
static microspeech_device_t* device = NULL;

static fifo_t sample_fifo;

static int buf_num = 0;

microspeech_device_t* get_microspeech_device() {
    return device;
}

void mic_decoupler(chanend_t c_ds_output, chanend_t c_gpio) {
    fifo_init(sample_fifo, MIC_FRAME_BUFFER_COUNT-1,
              sizeof(int32_t), 1);

    mic_samples_ptr = &mic_samples[0][0];
    fifo_ptr = &sample_fifo;

    mic_device.sample_fifo = fifo_ptr;
    mic_device.sample_buffer = mic_samples_ptr;

    device = &mic_device;

    TRIGGERABLE_SETUP_EVENT_VECTOR(c_ds_output, mic_data_ready);

    triggerable_disable_all();
    triggerable_enable_trigger(c_ds_output);

    for (;;) {

        TRIGGERABLE_WAIT_EVENT(mic_data_ready);

        mic_data_ready: {

            // on streaming chanend in;

            // int* unsafe mic_sample_block;
            // mic_sample_block = (int*)tmp;

            int* mic_sample_block = (int*)s_chan_in_word(c_ds_output);

            for(int i=0; i< (1 << MIC_ARRAY_MAX_FRAME_SIZE_LOG2); i++) {
                mic_samples[buf_num][i] = mic_sample_block[4*i];
            }

            fifo_put(sample_fifo, &buf_num);
            if (++buf_num == MIC_FRAME_BUFFER_COUNT) {
                buf_num = 0;
            }
            increment_timestamp( 16 );

            chan_out_word(c_gpio, get_led_status());

			continue;
        }
    }
}

void tile0(chanend_t c_gpio) {
  port_enable(p_gpio_leds);

  TRIGGERABLE_SETUP_EVENT_VECTOR(c_gpio, gpio_update);

  triggerable_disable_all();
  triggerable_enable_trigger(c_gpio);

  while(1) {
    TRIGGERABLE_WAIT_EVENT(gpio_update);

    gpio_update: {
      int tmp = chan_in_word(c_gpio);
      port_out(p_gpio_leds , tmp);
      continue;
    }
  }
}
