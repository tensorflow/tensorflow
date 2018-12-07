/*******************************************************************************
 *
 * Copyright (C) 2018 Eta Compute, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/


/*
 * linker script for use with ECM3531
 * All sections must map to 128KBytes of SRAM beginning at 0x10000000
 *
 */

 /*
  * Indicate to the linker the entry point.
  */
ENTRY(ResetISR)

/*
 *   SRAM is at 0x10000000 of length 0x00020000
 */
MEMORY
{
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
    } > SRAM= 0
    .dummy :
    {
        . = ALIGN(0x4);
        _eftext = .;
    } > SRAM
    .datax :
    {
        _datax = .;
        KEEP(*(.mainStack))
        . += 12288;
        _edatax = .;
        _stack_top = .;
        . += 4;
    } > SRAM
    .data :
       AT (ADDR(.text) + SIZEOF(.text) ) 
    {
        _data = .;
        *(.data*)
        KEEP(*(.mainHeap))
        _edata = .;
    } > SRAM

    .bss :
    {
        _bss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > SRAM
    .ARM.exidx :
    {
       *(.ARM.exidx*)
    }

}

