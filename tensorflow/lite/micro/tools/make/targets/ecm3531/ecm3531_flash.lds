/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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


/*
 * linker script for use with ECM3531 chip.
 * .text  and .ro map to FLASH all else to SRAM.
 *
 */

 /*
  * Indicate to the linker the entry point.
  */
ENTRY(ResetISR)

/*
 *   FLASH is at 0x01000000 of length 0x00080000  512KB
 *   SRAM  is at 0x10000000 of length 0x00020000  128KB
 */
MEMORY
{
    FLASH (RX) : ORIGIN = 0x01000000, LENGTH = 0x00080000
    SRAM (RWX) : ORIGIN = 0x10000000, LENGTH = 0x00020000
}

SECTIONS
{
    .text :
    {
        _text = .;
        KEEP(*(.vectors))
        . = ALIGN(0x4);
        *(.text*)
        . = ALIGN(0x4);
        *(.rodata*)
        . = ALIGN(0x4);
        _etext = .;
    } > FLASH= 0
    .dummy :
    {
        . = ALIGN(0x4);
        _eftext = .;
    } > FLASH
/* put the stack at the bottom of SRAM*/
    .datax (NOLOAD) :
    {
        _datax = .;
        KEEP(*(.mainStack))
        . = ALIGN(0x4);
        . += 16384;
        _edatax = .;
        _stack_top = .;
    } > SRAM
    .data :
    {
        _data = .;
        *(.data*)
        KEEP(*(.mainHeap))
        _edata = .;
    } > SRAM AT > FLASH

    .bss (NOLOAD) :
    {
        _bss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > SRAM



}

