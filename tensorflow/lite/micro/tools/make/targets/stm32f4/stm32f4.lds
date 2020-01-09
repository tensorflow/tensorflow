/* Copyright 2020 Google Inc. All Rights Reserved.

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

/* Copied and modified from: tensorflow/lite/micro/tools/make/targets/bluepill/bluepill.lds

*/

/*
 * 0x00000000 - 0x07ffffff - aliased to flash or sys memory depending on BOOT jumpers.
 * 0x08000000 - 0x0801ffff - Flash.
 * 0x1ffff000 - 0x1ffff7ff - Boot firmware in system memory.
 * 0x1ffff800 - 0x1fffffff - Option bytes.
 * 0x20000000 - 0x20004fff - SRAM.
 * 0x40000000 - 0x40023400 - Peripherals
 */

/* Define main entry point */
ENTRY(_main)

/* 20K of RAM and 128K of FLASH */
MEMORY {
RAM (xrw) : ORIGIN = 0x20000000, LENGTH = 20K
FLASH (rx) : ORIGIN = 0x8000000, LENGTH = 256K
}

/* Compute where the stack ends rather than hard coding it */
_ld_stack_end_addr = ORIGIN(RAM) + LENGTH(RAM);
_ld_min_stack_size = 0x200;

SECTIONS {

/* interrupt vector goes to top of flash */

.interrupt_vector : {
  . = ALIGN(4);
  KEEP(*(.interrupt_vector))
  . = ALIGN(4);
} >FLASH

/* read only .text and .rodata go to flash */

.text : {
  . = ALIGN(4);
  KEEP(*(.text.interrupt_handler))
  *(.text*)
} >FLASH

.rodata : {
  . = ALIGN(4);
  *(.rodata*)
  . = ALIGN(4);
} >FLASH

/* read mwrite data needs to be stored in flash but copied to ram */
.data : {
  . = ALIGN(4);
  _ld_data_load_dest_start = .; /* export where to load from */
  *(.data*)
  . = ALIGN(4);
  _ld_data_load_dest_stop = .; /* export where to load from */
} >RAM AT> FLASH
_ld_data_load_source = LOADADDR(.data);

/* unitialized data section needs zero initialization */
.bss :
{
  . = ALIGN(4);
  _ld_bss_data_start = .;
  *(.bss*)
  . = ALIGN(4);
  _ld_bss_data_stop = .;
} >RAM

._user_heap_stack :
{
  . = ALIGN(8);
  . += _ld_min_stack_size;
  PROVIDE(end = .);
  . = ALIGN(8);
} >RAM

/DISCARD/ :
{
  libc.a (*)
  libm.a (*)
  libgcc.a (*)
}

}  /* SECTIONS */
