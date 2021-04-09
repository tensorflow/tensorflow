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


OUTPUT(a.elf)

/* By default, program starts from reset address (the default location of the interrupt table) */
ENTRY(__cxd_inttbl_start)

/** Memory configuration parameters.
 *  The parameters become application symbols and can be referred from application
 */
__internal_data_start = DEFINED(__internal_data_start) ? __internal_data_start : 0x00000000;
__internal_data_size  = DEFINED(__internal_data_size ) ? __internal_data_size  : 256k;
__external_start      = DEFINED(__external_start     ) ? __external_start      : 0x20000000;
__external_size       = DEFINED(__external_size      ) ? __external_size       : 0x60000000;
__rom_start           = DEFINED(__rom_start          ) ? __rom_start           : 0xC0000000;
__rom_size            = DEFINED(__rom_size           ) ? __rom_size            : 1024M;

__malloc_size         = DEFINED(__malloc_size        ) ? __malloc_size         : 16k;
__stack_size          = DEFINED(__stack_size         ) ? __stack_size          : 16k;
__arg_sect_size       = DEFINED(__arg_sect_size      ) ? __arg_sect_size       : 512;

MEMORY {
    INTERNAL_DATA  (rw) : ORIGIN = __internal_data_start, LENGTH = __internal_data_size
    EXTERNAL      (rwx) : ORIGIN = __external_start     , LENGTH = __external_size
    ROM            (rx) : ORIGIN = __rom_start          , LENGTH = __rom_size
}

SECTIONS {
    .inttbl : ALIGN(0x20) {
        /** The interrupt vector table. Contains the NMI
         *  and maskable interrupt handlers
         */
        . = 0x0;
        KEEP(*(.inttbl))
        . = ALIGN(0x20);
    	KEEP(*(.sinttbl))
    } >EXTERNAL

    .data.internal : ALIGN(0x20) {
        PROVIDE(__data_internal_start = ABSOLUTE(.));
        /* Don't map any data at address zero to avoid issues with C NULL
         * pointer checks
         */
        . += 0x4;

        PROVIDE(__data_start = ABSOLUTE(.));
        *(.data .data.*)
        PROVIDE(__data_end = ABSOLUTE(.));
        PROVIDE(__data_size = ABSOLUTE(+__data_end - __data_start));

        PROVIDE(__sdata_start = ABSOLUTE(.));
        *(.sdata .sdata.*)
        PROVIDE(__sdata_end = ABSOLUTE(.));
        PROVIDE(__sdata_size = ABSOLUTE(+__sdata_end - __sdata_start));

        PROVIDE(__data_internal_end = ABSOLUTE(.));
        PROVIDE(__data_internal_size = ABSOLUTE(__data_internal_end - __data_internal_start));
    } >INTERNAL_DATA

    .data.internal.clone (NOLOAD) : ALIGN(0x20) {
  		PROVIDE(__data_internal_clone_start = ABSOLUTE(.));
		. = ABSOLUTE(. + __data_internal_size);
  	} >INTERNAL_DATA

    .data.internal.ro : ALIGN(0x20) {
        PROVIDE(__data_internal_ro_start = ABSOLUTE(.));
        PROVIDE(__rodata_start = ABSOLUTE(.));
        *(.rodata .rodata.*)
        PROVIDE(__rodata_end = ABSOLUTE(.));
        PROVIDE(__rodata_size = ABSOLUTE(+__rodata_end - __rodata_start));

        PROVIDE(__data_internal_ro_end = ABSOLUTE(.));
        PROVIDE(__data_internal_ro_size = ABSOLUTE(__data_internal_ro_end - __data_internal_ro_start));
    } >INTERNAL_DATA

    .cst.call : ALIGN(4) {
        PROVIDE(__cst_call_start = ABSOLUTE(.));
        *(.cst.call)
        PROVIDE(__cst_call_end = ABSOLUTE(.));
    } >INTERNAL_DATA

    .cst.mov : ALIGN(4) {
        PROVIDE(__cst_mov_start = ABSOLUTE(.));
        *(.cst.mov)
        PROVIDE(__cst_mov_end = ABSOLUTE(.));
    } >INTERNAL_DATA

    .bss (NOLOAD) : ALIGN(0x20) {
        PROVIDE(__bss_start = ABSOLUTE(.));
        *(.bss .bss.*)
        PROVIDE(__common_start = ABSOLUTE(.));
        *(COMMON)
        PROVIDE(__common_end = ABSOLUTE(.));
        PROVIDE(__common_size = ABSOLUTE(+__common_end - __common_start));
        PROVIDE(__bss_end = ABSOLUTE(.));
        PROVIDE(__bss_size = ABSOLUTE(+__bss_end - __bss_start));
    } >INTERNAL_DATA

    __STACK_SECT (NOLOAD) : ALIGN(0x10) {
        __stack_start = ABSOLUTE(.);
        . = . + __stack_size;
        __stack_end = ABSOLUTE(.);
    } >INTERNAL_DATA

    .text : ALIGN(0x20) {
        PROVIDE(__text_start = ABSOLUTE(.));
        /* The __call_saved* functions need to be placed at low addresses for
         * calling with absolute call instructions
         */
        *(.text.__call_saved*)
        *(.text .text.*)
		/* Program sections in external memory should be aligned to the fetch line width
		*/
		. = ALIGN(0x20);
        PROVIDE(__text_end = ABSOLUTE(.));
    } >EXTERNAL

    .data.external : ALIGN(0x20) {
        /** .data1, .rodata1, .sdata1 are all for large symbols which cannot
         * fit in limited internal memory. We put them in external memory by
         * default. */
        PROVIDE(__data_external_start = ABSOLUTE(.));

        PROVIDE(__data1_start = ABSOLUTE(.));
        *(.data1 .data1.*)
        PROVIDE(__data1_end = ABSOLUTE(.));

        PROVIDE(__sdata1_start = ABSOLUTE(.));
        *(.sdata1 .sdata1.*)
        PROVIDE(__sdata1_end = ABSOLUTE(.));
        PROVIDE(__sdata1_size = ABSOLUTE(+__sdata1_end - __sdata1_start));

        PROVIDE(__data_external_end = ABSOLUTE(.));
        PROVIDE(__data_external_size = ABSOLUTE(__data_external_end - __data_external_start));
    } >EXTERNAL

    .data.external.clone (NOLOAD) : ALIGN(0x20) {
        PROVIDE(__data_external_clone_start = ABSOLUTE(.));
		. = ABSOLUTE(. + __data_external_size);
   } >EXTERNAL

    .data.external.ro : ALIGN(0x20) {
        /** .data1, .rodata1, .sdata1 are all for large symbols which cannot
         * fit in limited internal memory. We put them in external memory by
         * default. */
        PROVIDE(__data_external_ro_start = ABSOLUTE(.));

        PROVIDE(__rodata1_start = ABSOLUTE(.));
        *(.rodata1 .rodata1.*)
        PROVIDE(__rodata1_end = ABSOLUTE(.));
        PROVIDE(__rodata1_size = ABSOLUTE(+__rodata1_end - __rodata1_start));

        /* Constructors and destructors are called once per program invocation,
         * so are never in the hot path; they shouldn't waste space in limited
         * internal memory so we place them in slower, external memory */

        . = ALIGN(4); /* constructors must be aligned on a word boundary */
        PROVIDE(__init_array_start = ABSOLUTE(.));
        KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)));
        KEEP(*(SORT_BY_INIT_PRIORITY(.init_array*) SORT_BY_INIT_PRIORITY(.ctors*)));
        PROVIDE(__init_array_end = ABSOLUTE(.));

        PROVIDE(__fini_array_start = ABSOLUTE(.));
        /* destructors are run in reverse order of their priority */
        KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)));
        KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array*) SORT_BY_INIT_PRIORITY(.dtors*)));
        PROVIDE(__fini_array_end = ABSOLUTE(.));

        PROVIDE(__data_external_ro_end = ABSOLUTE(.));
        PROVIDE(__data_external_ro_size = ABSOLUTE(__data_external_ro_end - __data_external_ro_start));
    } >EXTERNAL

    .bss1 (NOLOAD) : ALIGN(0x20) {
        /**
         * `.bss1` is for large zero-initialized symbols that do not fit in
         * internal data
         */
        PROVIDE(__bss1_start = ABSOLUTE(.));
        *(.bss1 .bss1.*)
        PROVIDE(__large_common_start = ABSOLUTE(.));
        *(LARGE_COMMON)
        PROVIDE(__large_common_end = ABSOLUTE(.));
        PROVIDE(__large_common_size = ABSOLUTE(+__large_common_end - __large_common_start));
        PROVIDE(__bss1_end = ABSOLUTE(.));
        PROVIDE(__bss1_size = ABSOLUTE(+__bss1_end - __bss1_start));
    } >EXTERNAL

    /* Program arguments are loaded by `_start` routine from `__arg_sect_start`.
     * When the user has set a zero size for the section, argc, and argv
     * will be zero and NULL, respectively.
     * Although likely small, they are on the slow path so by default they
     * go at the end of external memory
     */
    __ARG_SECT (NOLOAD) : ALIGN(0x4) {
        __arg_sect_start = .;
        . = . + (__arg_sect_size ? __arg_sect_size + 4 : 0);
        __arg_sect_end = .;
    } >EXTERNAL

    __MALLOC_SECT (NOLOAD) : ALIGN(0x10) {
        PROVIDE(__malloc_start = ABSOLUTE(.));
        . = . + __malloc_size;
        PROVIDE(__malloc_end = ABSOLUTE(.));
    } >EXTERNAL

    data_internal_loadable_addr = __data_internal_clone_start;
    data_external_loadable_addr = __data_external_clone_start;

    /DISCARD/ : {
        /* Note:  The CEVA Debugger and Restriction Checker use information 
         * stored in the ".note.CEVA-arch" section. Do NOT discard this section
         * for projects in development phase. This section has no effect on the
         * applications footprint */
        *(.comment)
        *(.note.GNU-stack)
        /* The X-DSP ABI uses a custom relocation format stored in its own
         * section. These are left in the binary by default but are unneeded. */
        *(.ceva_reloc)
    }

}
