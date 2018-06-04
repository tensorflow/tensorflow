# Description:
#   libjpeg-turbo is a drop in replacement for jpeglib optimized with SIMD.

licenses(["notice"])  # custom notice-style license, see LICENSE.md

exports_files(["LICENSE.md"])

load("@org_tensorflow//third_party:common.bzl", "template_rule")

libjpegturbo_nocopts = "-[W]error"

WIN_COPTS = [
    "/Ox",
    "/w14711",  # function 'function' selected for inline expansion
    "/w14710",  # 'function' : function not inlined
]

libjpegturbo_copts = select({
    ":android": [
        "-O2",
        "-fPIE",
        "-w",
    ],
    ":windows": WIN_COPTS,
    ":windows_msvc": WIN_COPTS,
    "//conditions:default": [
        "-O3",
        "-w",
    ],
}) + select({
    ":armeabi-v7a": [
        "-D__ARM_NEON__",
        "-march=armv7-a",
        "-mfloat-abi=softfp",
        "-fprefetch-loop-arrays",
    ],
    ":linux_ppc64le": [
        "-mcpu=power8",
        "-mtune=power8",
    ],
    "//conditions:default": [],
})

cc_library(
    name = "jpeg",
    srcs = [
        "jaricom.c",
        "jcapimin.c",
        "jcapistd.c",
        "jcarith.c",
        "jccoefct.c",
        "jccolor.c",
        "jcdctmgr.c",
        "jchuff.c",
        "jchuff.h",
        "jcinit.c",
        "jcmainct.c",
        "jcmarker.c",
        "jcmaster.c",
        "jcomapi.c",
        "jconfig.h",
        "jconfigint.h",
        "jcparam.c",
        "jcphuff.c",
        "jcprepct.c",
        "jcsample.c",
        "jctrans.c",
        "jdapimin.c",
        "jdapistd.c",
        "jdarith.c",
        "jdatadst.c",
        "jdatasrc.c",
        "jdcoefct.c",
        "jdcoefct.h",
        "jdcolor.c",
        "jdct.h",
        "jddctmgr.c",
        "jdhuff.c",
        "jdhuff.h",
        "jdinput.c",
        "jdmainct.c",
        "jdmainct.h",
        "jdmarker.c",
        "jdmaster.c",
        "jdmaster.h",
        "jdmerge.c",
        "jdphuff.c",
        "jdpostct.c",
        "jdsample.c",
        "jdsample.h",
        "jdtrans.c",
        "jerror.c",
        "jfdctflt.c",
        "jfdctfst.c",
        "jfdctint.c",
        "jidctflt.c",
        "jidctfst.c",
        "jidctint.c",
        "jidctred.c",
        "jinclude.h",
        "jmemmgr.c",
        "jmemnobs.c",
        "jmemsys.h",
        "jpeg_nbits_table.h",
        "jpegcomp.h",
        "jquant1.c",
        "jquant2.c",
        "jutils.c",
        "jversion.h",
    ],
    hdrs = [
        "jccolext.c",  # should have been named .inc
        "jdcol565.c",  # should have been named .inc
        "jdcolext.c",  # should have been named .inc
        "jdmrg565.c",  # should have been named .inc
        "jdmrgext.c",  # should have been named .inc
        "jerror.h",
        "jmorecfg.h",
        "jpegint.h",
        "jpeglib.h",
        "jstdhuff.c",  # should have been named .inc
    ],
    copts = libjpegturbo_copts,
    nocopts = libjpegturbo_nocopts,
    visibility = ["//visibility:public"],
    deps = select({
        ":k8": [":simd_x86_64"],
        ":armeabi-v7a": [":simd_armv7a"],
        ":arm64-v8a": [":simd_armv8a"],
        ":linux_ppc64le": [":simd_altivec"],
        "//conditions:default": [":simd_none"],
    }),
)

cc_library(
    name = "simd_altivec",
    srcs = [
        "jchuff.h",
        "jconfig.h",
        "jdct.h",
        "jerror.h",
        "jinclude.h",
        "jmorecfg.h",
        "jpegint.h",
        "jpeglib.h",
        "jsimd.h",
        "jsimddct.h",
        "simd/jccolor-altivec.c",
        "simd/jcgray-altivec.c",
        "simd/jcsample.h",
        "simd/jcsample-altivec.c",
        "simd/jdcolor-altivec.c",
        "simd/jdmerge-altivec.c",
        "simd/jdsample-altivec.c",
        "simd/jfdctfst-altivec.c",
        "simd/jfdctint-altivec.c",
        "simd/jidctfst-altivec.c",
        "simd/jidctint-altivec.c",
        "simd/jquanti-altivec.c",
        "simd/jsimd.h",
        "simd/jsimd_altivec.h",
        "simd/jsimd_powerpc.c",
    ],
    hdrs = [
        "simd/jccolext-altivec.c",  # should have been named .inc
        "simd/jcgryext-altivec.c",  # should have been named .inc
        "simd/jdcolext-altivec.c",  # should have been named .inc
        "simd/jdmrgext-altivec.c",  # should have been named .inc
    ],
    copts = libjpegturbo_copts,
    nocopts = libjpegturbo_nocopts,
)

cc_library(
    name = "simd_x86_64",
    srcs = [
        "jchuff.h",
        "jconfig.h",
        "jdct.h",
        "jerror.h",
        "jinclude.h",
        "jmorecfg.h",
        "jpegint.h",
        "jpeglib.h",
        "jsimd.h",
        "jsimddct.h",
        "simd/jccolor-sse2-64.o",
        "simd/jcgray-sse2-64.o",
        "simd/jchuff-sse2-64.o",
        "simd/jcsample-sse2-64.o",
        "simd/jdcolor-sse2-64.o",
        "simd/jdmerge-sse2-64.o",
        "simd/jdsample-sse2-64.o",
        "simd/jfdctflt-sse-64.o",
        "simd/jfdctfst-sse2-64.o",
        "simd/jfdctint-sse2-64.o",
        "simd/jidctflt-sse2-64.o",
        "simd/jidctfst-sse2-64.o",
        "simd/jidctint-sse2-64.o",
        "simd/jidctred-sse2-64.o",
        "simd/jquantf-sse2-64.o",
        "simd/jquanti-sse2-64.o",
        "simd/jsimd.h",
        "simd/jsimd_x86_64.c",
    ],
    copts = libjpegturbo_copts,
    linkstatic = 1,
    nocopts = libjpegturbo_nocopts,
)

genrule(
    name = "simd_x86_64_assemblage23",
    srcs = [
        "simd/jccolext-sse2-64.asm",
        "simd/jccolor-sse2-64.asm",
        "simd/jcgray-sse2-64.asm",
        "simd/jcgryext-sse2-64.asm",
        "simd/jchuff-sse2-64.asm",
        "simd/jcolsamp.inc",
        "simd/jcsample-sse2-64.asm",
        "simd/jdcolext-sse2-64.asm",
        "simd/jdcolor-sse2-64.asm",
        "simd/jdct.inc",
        "simd/jdmerge-sse2-64.asm",
        "simd/jdmrgext-sse2-64.asm",
        "simd/jdsample-sse2-64.asm",
        "simd/jfdctflt-sse-64.asm",
        "simd/jfdctfst-sse2-64.asm",
        "simd/jfdctint-sse2-64.asm",
        "simd/jidctflt-sse2-64.asm",
        "simd/jidctfst-sse2-64.asm",
        "simd/jidctint-sse2-64.asm",
        "simd/jidctred-sse2-64.asm",
        "simd/jpeg_nbits_table.inc",
        "simd/jquantf-sse2-64.asm",
        "simd/jquanti-sse2-64.asm",
        "simd/jsimdcfg.inc",
        "simd/jsimdext.inc",
    ],
    outs = [
        "simd/jccolor-sse2-64.o",
        "simd/jcgray-sse2-64.o",
        "simd/jchuff-sse2-64.o",
        "simd/jcsample-sse2-64.o",
        "simd/jdcolor-sse2-64.o",
        "simd/jdmerge-sse2-64.o",
        "simd/jdsample-sse2-64.o",
        "simd/jfdctflt-sse-64.o",
        "simd/jfdctfst-sse2-64.o",
        "simd/jfdctint-sse2-64.o",
        "simd/jidctflt-sse2-64.o",
        "simd/jidctfst-sse2-64.o",
        "simd/jidctint-sse2-64.o",
        "simd/jidctred-sse2-64.o",
        "simd/jquantf-sse2-64.o",
        "simd/jquanti-sse2-64.o",
    ],
    cmd = "for out in $(OUTS); do\n" +
          "  $(location @nasm//:nasm) -f elf64" +
          "    -DELF -DPIC -DRGBX_FILLER_0XFF -D__x86_64__ -DARCH_X86_64" +
          "    -I $$(dirname $(location simd/jdct.inc))/" +
          "    -I $$(dirname $(location simd/jsimdcfg.inc))/" +
          "    -o $$out" +
          "    $$(dirname $(location simd/jdct.inc))/$$(basename $${out%.o}.asm)\n" +
          "done",
    tools = ["@nasm"],
)

cc_library(
    name = "simd_armv7a",
    srcs = [
        "jchuff.h",
        "jconfig.h",
        "jdct.h",
        "jerror.h",
        "jinclude.h",
        "jmorecfg.h",
        "jpegint.h",
        "jpeglib.h",
        "jsimd.h",
        "jsimddct.h",
        "simd/jsimd.h",
        "simd/jsimd_arm.c",
        "simd/jsimd_arm_neon.S",
    ],
    copts = libjpegturbo_copts,
    nocopts = libjpegturbo_nocopts,
)

cc_library(
    name = "simd_armv8a",
    srcs = [
        "jchuff.h",
        "jconfig.h",
        "jdct.h",
        "jerror.h",
        "jinclude.h",
        "jmorecfg.h",
        "jpegint.h",
        "jpeglib.h",
        "jsimd.h",
        "jsimddct.h",
        "simd/jsimd.h",
        "simd/jsimd_arm64.c",
        "simd/jsimd_arm64_neon.S",
    ],
    copts = libjpegturbo_copts,
    nocopts = libjpegturbo_nocopts,
)

cc_library(
    name = "simd_none",
    srcs = [
        "jchuff.h",
        "jconfig.h",
        "jdct.h",
        "jerror.h",
        "jinclude.h",
        "jmorecfg.h",
        "jpegint.h",
        "jpeglib.h",
        "jsimd.h",
        "jsimd_none.c",
        "jsimddct.h",
    ],
    copts = libjpegturbo_copts,
    nocopts = libjpegturbo_nocopts,
)

template_rule(
    name = "jconfig_win",
    src = "win/jconfig.h.in",
    out = "jconfig_win.h",
    substitutions = {
        "@JPEG_LIB_VERSION@": "62",
        "@VERSION@": "1.5.1",
        "@LIBJPEG_TURBO_VERSION_NUMBER@": "1005001",
        "cmakedefine": "define",
        "@BITS_IN_JSAMPLE@": "8",
    },
)

template_rule(
    name = "jconfigint_win",
    src = "win/jconfigint.h.in",
    out = "jconfigint_win.h",
    substitutions = {
        "@VERSION@": "1.5.1",
        "@BUILD@": "20161115",
        "@CMAKE_PROJECT_NAME@": "libjpeg-turbo",
    },
)

JCONFIG_NOWIN_COMMON_SUBSTITUTIONS = {
    "LIBJPEG_TURBO_VERSION 0": "LIBJPEG_TURBO_VERSION 1.5.1",
    "LIBJPEG_TURBO_VERSION_NUMBER 0": "LIBJPEG_TURBO_VERSION_NUMBER 1005001",
    "#undef C_ARITH_CODING_SUPPORTED": "#define C_ARITH_CODING_SUPPORTED 1",
    "#undef D_ARITH_CODING_SUPPORTED": "#define D_ARITH_CODING_SUPPORTED 1",
    "#undef HAVE_LOCALE_H": "#define HAVE_LOCALE_H 1",
    "#undef HAVE_STDDEF_H": "#define HAVE_STDDEF_H 1",
    "#undef HAVE_STDLIB_H": "#define HAVE_STDLIB_H 1",
    "#undef HAVE_UNSIGNED_CHAR": "#define HAVE_UNSIGNED_CHAR 1",
    "#undef HAVE_UNSIGNED_SHORT": "#define HAVE_UNSIGNED_SHORT 1",
    "#undef INCOMPLETE_TYPES_BROKEN": "",
    "#undef MEM_SRCDST_SUPPORTED": "#define MEM_SRCDST_SUPPORTED 1",
    "#undef NEED_BSD_STRINGS": "",
    "#undef NEED_SYS_TYPES_H": "#define NEED_SYS_TYPES_H 1",
    "#undef __CHAR_UNSIGNED__": "",
    "#undef const": "",
    "#undef size_t": "",
    "#undef RIGHT_SHIFT_IS_UNSIGNED": "",
}

JCONFIG_NOWIN_SIMD_SUBSTITUTIONS = {
    "#undef WITH_SIMD": "#define WITH_SIMD 1",
}

JCONFIG_NOWIN_NOSIMD_SUBSTITUTIONS = {
    "#undef WITH_SIMD": "",
}

JCONFIG_NOWIN_SIMD_SUBSTITUTIONS.update(JCONFIG_NOWIN_COMMON_SUBSTITUTIONS)

JCONFIG_NOWIN_NOSIMD_SUBSTITUTIONS.update(JCONFIG_NOWIN_COMMON_SUBSTITUTIONS)

template_rule(
    name = "jconfig_nowin_nosimd",
    src = "jconfig.h.in",
    out = "jconfig_nowin_nosimd.h",
    substitutions = JCONFIG_NOWIN_NOSIMD_SUBSTITUTIONS,
)

template_rule(
    name = "jconfig_nowin_simd",
    src = "jconfig.h.in",
    out = "jconfig_nowin_simd.h",
    substitutions = JCONFIG_NOWIN_SIMD_SUBSTITUTIONS,
)

template_rule(
    name = "jconfigint_nowin",
    src = "jconfigint.h.in",
    out = "jconfigint_nowin.h",
    substitutions = {
        "#undef BUILD": "#define BUILD \"20161115\"",
        "#undef inline": "",
        "#undef INLINE": "#define INLINE inline __attribute__((always_inline))",
        "#undef PACKAGE_NAME": "#define PACKAGE_NAME \"libjpeg-turbo\"",
        "#undef VERSION": "#define VERSION \"1.5.1\"",
        "#undef SIZEOF_SIZE_T": "#if (__WORDSIZE==64 && !defined(__native_client__))\n" +
                                "#define SIZEOF_SIZE_T 8\n" +
                                "#else\n" +
                                "#define SIZEOF_SIZE_T 4\n" +
                                "#endif\n",
    },
)

genrule(
    name = "configure",
    srcs = [
        "jconfig_win.h",
        "jconfig_nowin_nosimd.h",
        "jconfig_nowin_simd.h",
    ],
    outs = ["jconfig.h"],
    cmd = select({
        ":windows": "cp $(location jconfig_win.h) $@",
        ":windows_msvc": "cp $(location jconfig_win.h) $@",
        ":k8": "cp $(location jconfig_nowin_simd.h) $@",
        ":armeabi-v7a": "cp $(location jconfig_nowin_simd.h) $@",
        ":arm64-v8a": "cp $(location jconfig_nowin_simd.h) $@",
        ":linux_ppc64le": "cp $(location jconfig_nowin_simd.h) $@",
        "//conditions:default": "cp $(location jconfig_nowin_nosimd.h) $@",
    }),
)

genrule(
    name = "configure_internal",
    srcs = [
        "jconfigint_win.h",
        "jconfigint_nowin.h",
    ],
    outs = ["jconfigint.h"],
    cmd = select({
        ":windows": "cp $(location jconfigint_win.h) $@",
        ":windows_msvc": "cp $(location jconfigint_win.h) $@",
        "//conditions:default": "cp $(location jconfigint_nowin.h) $@",
    }),
)

# jiminy cricket the way this file is generated is completely outrageous
genrule(
    name = "configure_simd",
    outs = ["simd/jsimdcfg.inc"],
    cmd = "cat <<'EOF' >$@\n" +
          "%define DCTSIZE 8\n" +
          "%define DCTSIZE2 64\n" +
          "%define RGB_RED 0\n" +
          "%define RGB_GREEN 1\n" +
          "%define RGB_BLUE 2\n" +
          "%define RGB_PIXELSIZE 3\n" +
          "%define EXT_RGB_RED 0\n" +
          "%define EXT_RGB_GREEN 1\n" +
          "%define EXT_RGB_BLUE 2\n" +
          "%define EXT_RGB_PIXELSIZE 3\n" +
          "%define EXT_RGBX_RED 0\n" +
          "%define EXT_RGBX_GREEN 1\n" +
          "%define EXT_RGBX_BLUE 2\n" +
          "%define EXT_RGBX_PIXELSIZE 4\n" +
          "%define EXT_BGR_RED 2\n" +
          "%define EXT_BGR_GREEN 1\n" +
          "%define EXT_BGR_BLUE 0\n" +
          "%define EXT_BGR_PIXELSIZE 3\n" +
          "%define EXT_BGRX_RED 2\n" +
          "%define EXT_BGRX_GREEN 1\n" +
          "%define EXT_BGRX_BLUE 0\n" +
          "%define EXT_BGRX_PIXELSIZE 4\n" +
          "%define EXT_XBGR_RED 3\n" +
          "%define EXT_XBGR_GREEN 2\n" +
          "%define EXT_XBGR_BLUE 1\n" +
          "%define EXT_XBGR_PIXELSIZE 4\n" +
          "%define EXT_XRGB_RED 1\n" +
          "%define EXT_XRGB_GREEN 2\n" +
          "%define EXT_XRGB_BLUE 3\n" +
          "%define EXT_XRGB_PIXELSIZE 4\n" +
          "%define RGBX_FILLER_0XFF 1\n" +
          "%define JSAMPLE byte ; unsigned char\n" +
          "%define SIZEOF_JSAMPLE SIZEOF_BYTE ; sizeof(JSAMPLE)\n" +
          "%define CENTERJSAMPLE 128\n" +
          "%define JCOEF word ; short\n" +
          "%define SIZEOF_JCOEF SIZEOF_WORD ; sizeof(JCOEF)\n" +
          "%define JDIMENSION dword ; unsigned int\n" +
          "%define SIZEOF_JDIMENSION SIZEOF_DWORD ; sizeof(JDIMENSION)\n" +
          "%define JSAMPROW POINTER ; JSAMPLE * (jpeglib.h)\n" +
          "%define JSAMPARRAY POINTER ; JSAMPROW * (jpeglib.h)\n" +
          "%define JSAMPIMAGE POINTER ; JSAMPARRAY * (jpeglib.h)\n" +
          "%define JCOEFPTR POINTER ; JCOEF * (jpeglib.h)\n" +
          "%define SIZEOF_JSAMPROW SIZEOF_POINTER ; sizeof(JSAMPROW)\n" +
          "%define SIZEOF_JSAMPARRAY SIZEOF_POINTER ; sizeof(JSAMPARRAY)\n" +
          "%define SIZEOF_JSAMPIMAGE SIZEOF_POINTER ; sizeof(JSAMPIMAGE)\n" +
          "%define SIZEOF_JCOEFPTR SIZEOF_POINTER ; sizeof(JCOEFPTR)\n" +
          "%define DCTELEM word ; short\n" +
          "%define SIZEOF_DCTELEM SIZEOF_WORD ; sizeof(DCTELEM)\n" +
          "%define float FP32 ; float\n" +
          "%define SIZEOF_FAST_FLOAT SIZEOF_FP32 ; sizeof(float)\n" +
          "%define ISLOW_MULT_TYPE word ; must be short\n" +
          "%define SIZEOF_ISLOW_MULT_TYPE SIZEOF_WORD ; sizeof(ISLOW_MULT_TYPE)\n" +
          "%define IFAST_MULT_TYPE word ; must be short\n" +
          "%define SIZEOF_IFAST_MULT_TYPE SIZEOF_WORD ; sizeof(IFAST_MULT_TYPE)\n" +
          "%define IFAST_SCALE_BITS 2 ; fractional bits in scale factors\n" +
          "%define FLOAT_MULT_TYPE FP32 ; must be float\n" +
          "%define SIZEOF_FLOAT_MULT_TYPE SIZEOF_FP32 ; sizeof(FLOAT_MULT_TYPE)\n" +
          "%define JSIMD_NONE 0x00\n" +
          "%define JSIMD_MMX 0x01\n" +
          "%define JSIMD_3DNOW 0x02\n" +
          "%define JSIMD_SSE 0x04\n" +
          "%define JSIMD_SSE2 0x08\n" +
          "EOF",
)

config_setting(
    name = "k8",
    values = {"cpu": "k8"},
)

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
)

config_setting(
    name = "armeabi-v7a",
    values = {"cpu": "armeabi-v7a"},
)

config_setting(
    name = "arm64-v8a",
    values = {"cpu": "arm64-v8a"},
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
)

config_setting(
    name = "linux_ppc64le",
    values = {"cpu": "ppc"},
)
