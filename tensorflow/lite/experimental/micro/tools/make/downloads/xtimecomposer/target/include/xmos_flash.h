// Copyright (c) 2017-2018, XMOS Ltd, All rights reserved

#ifndef _XMOS_FLASH_H_
#define _XMOS_FLASH_H_

#if !defined(__XS1B__)

#include <xccompat.h> //From the xTIMEcomposer tools
#include <xs1.h>
#include <stddef.h>

typedef enum flash_status_register_t
{
  flash_status_register_0,
  flash_status_register_1,
  flash_status_register_2,
}flash_status_register_t;

typedef enum flash_num_status_bytes_t
{
  flash_num_status_bytes_1 = 1,
  flash_num_status_bytes_2 = 2,
  flash_num_status_bytes_3 = 3,
}flash_num_status_bytes_t;

typedef enum flash_clock_t
{
  flash_clock_reference,
  flash_clock_xcore,
}flash_clock_t;

typedef enum flash_clock_input_edge_t
{
  flash_clock_input_edge_rising,
  flash_clock_input_edge_falling,
  flash_clock_input_edge_plusone,
}flash_clock_input_edge_t;

typedef enum flash_port_pad_delay_t
{
  flash_port_pad_delay_0 = 0,
  flash_port_pad_delay_1 = 1,
  flash_port_pad_delay_2 = 2,
  flash_port_pad_delay_3 = 3,
  flash_port_pad_delay_4 = 4,
  flash_port_pad_delay_5 = 5,
}flash_port_pad_delay_t;

typedef enum flash_qe_location_t
{
  flash_qe_location_status_reg_0 = 0,
  flash_qe_location_status_reg_1 = 1,
}flash_qe_location_t;

typedef enum flash_qe_bit_t
{
  flash_qe_bit_0 = 0,
  flash_qe_bit_1 = 1,
  flash_qe_bit_2 = 2,
  flash_qe_bit_3 = 3,
  flash_qe_bit_4 = 4,
  flash_qe_bit_5 = 5,
  flash_qe_bit_6 = 6,
  flash_qe_bit_7 = 7,
}flash_qe_bit_t;

/**
 * Flash port structure
**/
typedef struct flash_ports_t
{
#ifdef __XC__
  out port                            flash_cs;
  out buffered port:32                flash_sclk;
  [[bidirectional]]buffered port:32   flash_sio;
  clock                               flash_clk_blk;
#else
  unsigned int                        flash_cs;
  unsigned int                        flash_sclk;
  unsigned int                        flash_sio;
  unsigned int                        flash_clk_blk;
#endif
} flash_ports_t;

/**
 * Flash clock configuration structure
**/
typedef struct flash_clock_config_t
{
  flash_clock_t                  flash_clock;
  unsigned int                   flash_input_clock_div;
  unsigned int                   flash_output_clock_div;
  flash_clock_input_edge_t       flash_port_input_edge;
  flash_port_pad_delay_t         flash_port_pad_delay;
} flash_clock_config_t;

/**
 * Flash quad enable bit configuration structure
**/
typedef struct flash_qe_config_t
{
  flash_qe_location_t  flash_qe_location;
  flash_qe_bit_t       flash_qe_shift;
} flash_qe_config_t;

/**
 * Flash handle
 **/
#define SIZEOF_FLASH_HANDLE 15
typedef struct flash_handle_t
{
  const unsigned x[SIZEOF_FLASH_HANDLE];
} flash_handle_t;

/**
 * flash_connect: connect to the quad spi device and configure.
 *
 * Note: The parameter flash_ports must be declared as a global.
 *
 *  \param flash_handle        The flash handle to be received from the library.
 *  \param flash_ports         The flash ports to be used by the library.
 *  \param flash_clock_config  The flash clock configuration.
 *  \param flash_qe_config     The flash quad enable bit location.
 *  \return                    1 if successfully connect, else 0.
 **/
int flash_connect(flash_handle_t * flash_handle,
                  const flash_ports_t * flash_ports,
                  flash_clock_config_t flash_clock_config,
                  flash_qe_config_t flash_qe_config);

/**
 * flash_disconnect: disconnect from the quad spi device
 *
 *  \param flash_handle    The flash handle obtined from flash_connect.
 **/
void flash_disconnect(const flash_handle_t * flash_handle);

/**
 * flash_read_jedec_id: read product identification by JEDEC id operation
 *
 *  \param flash_handle   The flash handle obtined from flash_connect.
 *  \param jedec          The buffer to hold the JEDEC id that is read.
 *  \param num_bytes      The number of bytes in the JEDEC id.
 **/
void flash_read_jedec_id(const flash_handle_t * flash_handle,
                         ARRAY_OF_SIZE(char, jedec, num_bytes),
                         size_t num_bytes);

/**
 * flash_write_enable: write enable operation
 *
 *  \param flash_handle    The flash handle obtined from flash_connect.
 **/
void flash_write_enable(const flash_handle_t * flash_handle);

/**
 * flash_write_disable: write disable operation
 *
 *  \param flash_handle    The flash handle obtined from flash_connect.
 **/
void flash_write_disable(const flash_handle_t * flash_handle);

/**
 * flash_read_status_register: read status register operation
 *
 *  \param flash_handle            The flash handle obtined from flash_connect.
 *  \param flash_status_register   The status register to be read from.
 *  \return                        The contents of the status register.
 **/
char flash_read_status_register(const flash_handle_t * flash_handle,
                                flash_status_register_t flash_status_register);

/**
 * flash_write_status_register: write status register operation
 *
 *  \param flash_handle       The flash handle obtined from flash_connect.
 *  \param status             The status register contents to be written.
 *  \param num_bytes          The number of bytes to be written to the status registers.
 **/
void flash_write_status_register(const flash_handle_t * flash_handle,
                                 ARRAY_OF_SIZE(const char, status, num_bytes),
                                 flash_num_status_bytes_t num_bytes);

/**
 * flash_read: read data operation in SPI mode (Max speed 31.25MHz)
 *
 *  \param flash_handle       The flash handle obtined from flash_connect.
 *  \param byte_address       The address to read from.
 *  \param destination        The buffer to store the data read.
 *  \param num_bytes          The number of bytes to read.
 **/
void flash_read(const flash_handle_t * flash_handle,
                unsigned byte_address,
                ARRAY_OF_SIZE(char, destination, num_bytes),
                size_t num_bytes);

/**
 * flash_read_fast: fast read data operation in SPI mode (Max speed 50MHz)
 *
 *  \param flash_handle       The flash handle obtined from flash_connect.
 *  \param byte_address       The address to read from.
 *  \param destination        The buffer to store the data read.
 *  \param num_bytes          The number of bytes to read.
 **/
void flash_read_fast(const flash_handle_t * flash_handle,
                     unsigned byte_address,
                     ARRAY_OF_SIZE(char, destination, num_bytes),
                     size_t num_bytes);

/**
 * flash_read_quad: quad read data operation in QSPI mode (Max speed 50MHz)
 *
 *  \param flash_handle       The flash handle obtined from flash_connect.
 *  \param word_address       The address to read from.
 *  \param destination        The buffer to store the data read.
 *  \param num_words          The number of words to read.
 **/
void flash_read_quad(const flash_handle_t * flash_handle,
                     unsigned word_address,
                     ARRAY_OF_SIZE(unsigned, destination, num_words),
                     size_t num_words);

/**
 * flash_write_page: write page operation
 *
 *  \param flash_handle       The flash handle obtined from flash_connect.
 *  \param byte_address       The address to write to.
 *  \param page               The data to be written.
 *  \param num_bytes          The number of bytes to write.
 **/
void flash_write_page(const flash_handle_t * flash_handle,
                      unsigned byte_address,
                      ARRAY_OF_SIZE(const char, page, num_bytes),
                      size_t num_bytes);

/**
 * flash_erase_sector: erase sector operation
 *
 *  \param flash_handle       The flash handle obtined from flash_connect.
 *  \param sector_address     The address to erase.
 **/
void flash_erase_sector(const flash_handle_t * flash_handle,
                        unsigned sector_address);

/**
 * flash_erase_block_32KB: erase 32KB block operation
 *
 *  \param flash_handle       The flash handle obtined from flash_connect.
 *  \param block_address      The address to erase.
 **/
void flash_erase_block_32KB(const flash_handle_t * flash_handle,
                            unsigned block_address);

/**
 * flash_erase_block_64KB: erase 32KB block operation
 *
 *  \param flash_handle       The flash handle obtined from flash_connect.
 *  \param block_address      The address to erase.
 **/
void flash_erase_block_64KB(const flash_handle_t * flash_handle,
                            unsigned block_address);

/**
 * flash_erase_chip: erase chip operation
 *
 *  \param flash_handle    The flash handle obtined from flash_connect.
 **/
void flash_erase_chip(const flash_handle_t * flash_handle);

#else //!defined(__XS1B__)
#error "xmos_flash.h may only be used for XS2 devices"
#endif //!defined(__XS1B__)

#endif //_XMOS_FLASH_H_
