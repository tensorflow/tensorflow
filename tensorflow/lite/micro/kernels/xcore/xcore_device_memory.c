// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "tensorflow/lite/micro/kernels/xcore/xcore_device_memory.h"

#include <stddef.h>
#include <string.h>

#ifdef XCORE

#include <xcore/port.h>
#include <xcore/swmem_fill.h>
#include <xmos_flash.h>

flash_ports_t flash_ports_0 = {PORT_SQI_CS, PORT_SQI_SCLK, PORT_SQI_SIO,
                               XS1_CLKBLK_5};

flash_clock_config_t flash_clock_config = {
    1, 8, 8, 1, 0,
};

flash_qe_config_t flash_qe_config_0 = {flash_qe_location_status_reg_0,
                                       flash_qe_bit_6};

flash_handle_t flash_handle;
swmem_fill_t swmem_fill_handle;

void swmem_fill(fill_slot_t address) {
  swmem_fill_buffer_t buf;
  unsigned int *buf_ptr = (unsigned int *)buf;

  flash_read_quad(&flash_handle, (address - (void *)XS1_SWMEM_BASE) >> 2,
                  buf_ptr, SWMEM_FILL_SIZE_WORDS);

  swmem_fill_populate_from_buffer(swmem_fill_handle, address, buf);
}

void swmem_setup() {
  flash_connect(&flash_handle, &flash_ports_0, flash_clock_config,
                flash_qe_config_0);

  swmem_fill_handle = swmem_fill_get();
}

void swmem_teardown() {
  swmem_fill_free(swmem_fill_handle);
  flash_disconnect(&flash_handle);
}

void swmem_handler(void *ignored) {
  fill_slot_t address = 0;
  while (1) {
    address = swmem_fill_in_address(swmem_fill_handle);
    swmem_fill(address);
  }
}

void memload(void **dest, void *src, size_t size) {
  if (IS_SWMEM(src)) {
    flash_read_quad(&flash_handle, ((uintptr_t)src - XS1_SWMEM_BASE) >> 2,
                    (unsigned int *)*dest, size);
  } else if (IS_EXTMEM(src)) {
    memcpy(*dest, src, size);
  }
}

#endif  // XCORE
