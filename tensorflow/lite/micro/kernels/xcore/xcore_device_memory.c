// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "tensorflow/lite/micro/kernels/xcore/xcore_device_memory.h"

#include <stddef.h>
#include <string.h>

#ifdef XCORE
#include <xcore/port.h>
#include <xcore/swmem_fill.h>
#include <xmos_flash.h>

#ifdef USE_SWMEM
#ifdef USE_QSPI_SWMEM_DEV
#include "qspi_flash_dev.h"
#include "soc.h"

static chanend_t swmem_c;

#define WORDS_TO_BYTES(w) ((w) * sizeof(uint32_t))
#define BYTES_TO_WORDS(b) (((b) + sizeof(uint32_t) - 1) / sizeof(uint32_t))

#define WORD_TO_BYTE_ADDRESS(w) WORDS_TO_BYTES(w)
#define BYTE_TO_WORD_ADDRESS(b) ((b) / sizeof(uint32_t))
#else
flash_ports_t flash_ports_0 = {PORT_SQI_CS, PORT_SQI_SCLK, PORT_SQI_SIO,
                               XS1_CLKBLK_5};

flash_clock_config_t flash_clock_config = {
    1, 8, 8, 1, 0,
};

flash_qe_config_t flash_qe_config_0 = {flash_qe_location_status_reg_0,
                                       flash_qe_bit_6};

flash_handle_t flash_handle;
#endif /* USE_QSPI_SWMEM_DEV */

swmem_fill_t swmem_fill_handle;

void swmem_fill(fill_slot_t address) {
  swmem_fill_buffer_t buf;
  unsigned int *buf_ptr = (unsigned int *)buf;

#ifdef USE_QSPI_SWMEM_DEV
  qspi_flash_dev_cmd_t local_cmd;
  local_cmd.operation = qspi_flash_dev_op_read;
  local_cmd.byte_address =
      WORD_TO_BYTE_ADDRESS((address - (void *)XS1_SWMEM_BASE) >> 2);
  local_cmd.byte_count = WORDS_TO_BYTES(SWMEM_FILL_SIZE_WORDS);

  soc_peripheral_function_code_tx(swmem_c, QSPI_DEV_SWMEM_REQ);
  soc_peripheral_varlist_tx(swmem_c, 1, sizeof(qspi_flash_dev_cmd_t),
                            &local_cmd);
  soc_peripheral_varlist_rx(swmem_c, 1, local_cmd.byte_count, buf_ptr);
  swmem_fill_populate_from_buffer(swmem_fill_handle, address, buf);
#else
  flash_read_quad(&flash_handle, (address - (void *)XS1_SWMEM_BASE) >> 2,
                  buf_ptr, SWMEM_FILL_SIZE_WORDS);

  uint32_t adr = (address - (void *)XS1_SWMEM_BASE) >> 2;

  swmem_fill_populate_from_buffer(swmem_fill_handle, address, buf);
#endif /* USE_QSPI_SWMEM_DEV */
}

#ifdef USE_QSPI_SWMEM_DEV
void swmem_setup(chanend_t ctrl_swmem_c) {
  swmem_c = ctrl_swmem_c;

  swmem_fill_handle = swmem_fill_get();
}
#else
void swmem_setup() {
  flash_connect(&flash_handle, &flash_ports_0, flash_clock_config,
                flash_qe_config_0);

  swmem_fill_handle = swmem_fill_get();
}
#endif /* USE_QSPI_SWMEM_DEV */

void swmem_teardown() {
  swmem_fill_free(swmem_fill_handle);
#ifdef USE_QSPI_SWMEM_DEV
#else
  flash_disconnect(&flash_handle);
#endif /* USE_QSPI_SWMEM_DEV */
}

void swmem_handler(void *ignored) {
  fill_slot_t address = 0;
  while (1) {
    address = swmem_fill_in_address(swmem_fill_handle);
    swmem_fill(address);
  }
}
#endif /* USE_SWMEM */

void memload(void *dest, void *src, size_t size) {
#ifdef USE_SWMEM
  if (IS_SWMEM(src)) {
#ifdef USE_QSPI_SWMEM_DEV
    qspi_flash_dev_cmd_t local_cmd;
    local_cmd.operation = qspi_flash_dev_op_read;
    local_cmd.byte_address =
        WORD_TO_BYTE_ADDRESS(((uintptr_t)src - XS1_SWMEM_BASE) >> 2);
    local_cmd.byte_count = WORDS_TO_BYTES(size);

    assert(local_cmd.byte_count <= QSPI_FLASH_DEV_WRITE_BUFSIZE);
    soc_peripheral_function_code_tx(swmem_c, QSPI_DEV_SWMEM_REQ);
    soc_peripheral_varlist_tx(swmem_c, 1, sizeof(qspi_flash_dev_cmd_t),
                              &local_cmd);
    soc_peripheral_varlist_rx(swmem_c, 1, size * sizeof(uint32_t *),
                              (unsigned int *)dest);
#else
    flash_read_quad(&flash_handle, ((uintptr_t)src - XS1_SWMEM_BASE) >> 2,
                    (unsigned int *)dest, size);
#endif /* USE_QSPI_SWMEM_DEV */
  } else
#endif /* USE_SWMEM */
      if (IS_EXTMEM(src)) {
    memcpy(dest, src, size);
  }
}

#endif  // XCORE
