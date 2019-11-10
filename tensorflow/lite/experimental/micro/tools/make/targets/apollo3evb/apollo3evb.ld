/******************************************************************************
 *
 * apollo3evb.ld - Linker script for applications using startup_gcc.c
 *
 *****************************************************************************/
ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx) : ORIGIN = 0x0000C000, LENGTH = 960K
    SRAM (rwx) : ORIGIN = 0x10000000, LENGTH = 384K
}

SECTIONS
{
    .text :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        KEEP(*(.patch))
        *(.text)
        *(.text*)

	/* These are the C++ global constructors.  Stick them all here and
	 * then walk through the array in main() calling them all.
	 */
	_init_array_start = .;
	KEEP (*(SORT(.init_array*)))
	_init_array_end = .;

	/* XXX Currently not doing anything for global destructors. */

        *(.rodata)
        *(.rodata*)
        . = ALIGN(4);
        _etext = .;
    } > FLASH

  /* User stack section initialized by startup code. */
  .stack (NOLOAD):
    {
        . = ALIGN(8);
        *(.stack)
        *(.stack*)
        . = ALIGN(8);
    } > SRAM

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > SRAM AT>FLASH

    /* used by startup to initialize data */
    _init_data = LOADADDR(.data);

    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > SRAM
    /* Add this to satisfy reference to symbol 'end' from libnosys.a(sbrk.o)
     * to denote the HEAP start.
     */
   end = .;

  .ARM.attributes 0 : { *(.ARM.attributes) }
}


