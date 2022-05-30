# hwloc: Portable Hardware Locality Library

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["COPYING"])

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@org_tensorflow//third_party:common.bzl", "template_rule")

COMMON_INCLUDE_COPTS = [
    "-I.",
    "-Ihwloc",
    "-Iinclude",
]

DISABLE_WARNINGS_COPTS = [
    "-Wno-vla",
]

VAR_SETTINGS_COPTS = [
    "-DHWLOC_DUMPED_HWDATA_DIR=",
    "-DRUNSTATEDIR=",
]

_INCLUDE_HWLOC_AUTOIGEN_CONFIG_H_COMMON_SUBS = {
    "#undef HWLOC_VERSION_MAJOR": "#define HWLOC_VERSION_MAJOR 2",
    "#undef HWLOC_VERSION_MINOR": "#define HWLOC_VERSION_MINOR 0",
    "#undef HWLOC_VERSION_RELEASE": "#define HWLOC_VERSION_RELEASE 3",
    "#undef HWLOC_VERSION_GREEK": "#define HWLOC_VERSION_GREEK \"\"",
    "#undef HWLOC_VERSION": "#define HWLOC_VERSION \"2.0.3\"",
    "#undef hwloc_pid_t": "#define hwloc_pid_t pid_t",
    "#undef hwloc_thread_t": "#define hwloc_thread_t pthread_t",
    "#  undef HWLOC_HAVE_STDINT_H": "#  define HWLOC_HAVE_STDINT_H 1",
    "#undef HWLOC_SYM_TRANSFORM": "#define HWLOC_SYM_TRANSFORM 0",
    "#undef HWLOC_SYM_PREFIX_CAPS": "#define HWLOC_SYM_PREFIX_CAPS HWLOC_",
    "#undef HWLOC_SYM_PREFIX": "#define HWLOC_SYM_PREFIX hwloc_",
}

_INCLUDE_HWLOC_AUTOIGEN_CONFIG_H_LINUX_SUBS = dict(_INCLUDE_HWLOC_AUTOIGEN_CONFIG_H_COMMON_SUBS)

_INCLUDE_HWLOC_AUTOIGEN_CONFIG_H_LINUX_SUBS.update({
    "#undef HWLOC_LINUX_SYS": "#define HWLOC_LINUX_SYS 1",
})

template_rule(
    name = "include_hwloc_autogen_config_h",
    src = "include/hwloc/autogen/config.h.in",
    out = "include/hwloc/autogen/config.h",
    substitutions = select({
        "@org_tensorflow//tensorflow:linux_x86_64": _INCLUDE_HWLOC_AUTOIGEN_CONFIG_H_LINUX_SUBS,
        "//conditions:default": _INCLUDE_HWLOC_AUTOIGEN_CONFIG_H_COMMON_SUBS,
    }),
)

_INCLUDE_PRIVATE_HWLOC_AUTOIGEN_CONFIG_H_COMMON_SUBS = {
    "#undef HAVE_CLOCK_GETTIME": "#define HAVE_CLOCK_GETTIME 1",
    "#undef HAVE_CTYPE_H": "#define HAVE_CTYPE_H 1",
    "#undef HAVE_DECL_CTL_HW": "#define HAVE_DECL_CTL_HW 0",
    "#undef HAVE_DECL_FABSF": "#define HAVE_DECL_FABSF 1",
    "#undef HAVE_DECL_GETEXECNAME": "#define HAVE_DECL_GETEXECNAME 0",
    "#undef HAVE_DECL_GETMODULEFILENAME": "#define HAVE_DECL_GETMODULEFILENAME 0",
    "#undef HAVE_DECL_GETPROGNAME": "#define HAVE_DECL_GETPROGNAME 0",
    "#undef HAVE_DECL_HW_NCPU": "#define HAVE_DECL_HW_NCPU 0",
    "#undef HAVE_DECL_MODFF": "#define HAVE_DECL_MODFF 1",
    "#undef HAVE_DECL_PTHREAD_GETAFFINITY_NP": "#define HAVE_DECL_PTHREAD_GETAFFINITY_NP 1",
    "#undef HAVE_DECL_PTHREAD_SETAFFINITY_NP": "#define HAVE_DECL_PTHREAD_SETAFFINITY_NP 1",
    "#undef HAVE_DECL_RUNNING_ON_VALGRIND": "#define HAVE_DECL_RUNNING_ON_VALGRIND 0",
    "#undef HAVE_DECL_SCHED_GETCPU": "#define HAVE_DECL_SCHED_GETCPU 1",
    "#undef HAVE_DECL_SNPRINTF": "#define HAVE_DECL_SNPRINTF 1",
    "#undef HAVE_DECL_STRTOULL": "#define HAVE_DECL_STRTOULL 1",
    "#undef HAVE_DECL__PUTENV": "#define HAVE_DECL__PUTENV 0",
    "#undef HAVE_DECL__SC_LARGE_PAGESIZE": "#define HAVE_DECL__SC_LARGE_PAGESIZE 0",
    "#undef HAVE_DECL__SC_NPROCESSORS_CONF": "#define HAVE_DECL__SC_NPROCESSORS_CONF 1",
    "#undef HAVE_DECL__SC_NPROCESSORS_ONLN": "#define HAVE_DECL__SC_NPROCESSORS_ONLN 1",
    "#undef HAVE_DECL__SC_NPROC_CONF": "#define HAVE_DECL__SC_NPROC_CONF 0",
    "#undef HAVE_DECL__SC_NPROC_ONLN": "#define HAVE_DECL__SC_NPROC_ONLN 0",
    "#undef HAVE_DECL__SC_PAGESIZE": "#define HAVE_DECL__SC_PAGESIZE 1",
    "#undef HAVE_DECL__SC_PAGE_SIZE": "#define HAVE_DECL__SC_PAGE_SIZE 1",
    "#undef HAVE_DECL__STRDUP": "#define HAVE_DECL__STRDUP 0",
    "#undef HAVE_DIRENT_H": "#define HAVE_DIRENT_H 1",
    "#undef HAVE_DLFCN_H": "#define HAVE_DLFCN_H 1",
    "#undef HAVE_FFSL": "#define HAVE_FFSL 1",
    "#undef HAVE_FFS": "#define HAVE_FFS 1",
    "#undef HAVE_GETPAGESIZE": "#define HAVE_GETPAGESIZE 1",
    "#undef HAVE_INTTYPES_H": "#define HAVE_INTTYPES_H 1",
    "#undef HAVE_LANGINFO_H": "#define HAVE_LANGINFO_H 1",
    "#undef HAVE_LOCALE_H": "#define HAVE_LOCALE_H 1",
    "#undef HAVE_MALLOC_H": "#define HAVE_MALLOC_H 1",
    "#undef HAVE_MEMALIGN": "#define HAVE_MEMALIGN 1",
    "#undef HAVE_MEMORY_H": "#define HAVE_MEMORY_H 1",
    "#undef HAVE_MKSTEMP": "#define HAVE_MKSTEMP 1",
    "#undef HAVE_NL_LANGINFO": "#define HAVE_NL_LANGINFO 1",
    "#undef HAVE_OPENAT": "#define HAVE_OPENAT 1",
    "#undef HAVE_POSIX_MEMALIGN": "#define HAVE_POSIX_MEMALIGN 1",
    "#undef HAVE_PTHREAD_T": "#define HAVE_PTHREAD_T 1",
    "#undef HAVE_PUTWC": "#define HAVE_PUTWC 1",
    "#undef HAVE_SETLOCALE": "#define HAVE_SETLOCALE 1",
    "#undef HAVE_SSIZE_T": "#define HAVE_SSIZE_T 1",
    "#undef HAVE_STDINT_H": "#define HAVE_STDINT_H 1",
    "#undef HAVE_STDLIB_H": "#define HAVE_STDLIB_H 1",
    "#undef HAVE_STRCASECMP": "#define HAVE_STRCASECMP 1",
    "#undef HAVE_STRFTIME": "#define HAVE_STRFTIME 1",
    "#undef HAVE_STRINGS_H": "#define HAVE_STRINGS_H 1",
    "#undef HAVE_STRING_H": "#define HAVE_STRING_H 1",
    "#undef HAVE_STRNCASECMP": "#define HAVE_STRNCASECMP 1",
    "#undef HAVE_SYS_MMAN_H": "#define HAVE_SYS_MMAN_H 1",
    "#undef HAVE_SYS_PARAM_H": "#define HAVE_SYS_PARAM_H 1",
    "#undef HAVE_SYS_STAT_H": "#define HAVE_SYS_STAT_H 1",
    "#undef HAVE_SYS_SYSCTL_H": "#define HAVE_SYS_SYSCTL_H 1",
    "#undef HAVE_SYS_TYPES_H": "#define HAVE_SYS_TYPES_H 1",
    "#undef HAVE_SYS_UTSNAME_H": "#define HAVE_SYS_UTSNAME_H 1",
    "#undef HAVE_TIME_H": "#define HAVE_TIME_H 1",
    "#undef HAVE_UNAME": "#define HAVE_UNAME 1",
    "#undef HAVE_UNISTD_H": "#define HAVE_UNISTD_H 1",
    "#undef HAVE_USELOCALE": "#define HAVE_USELOCALE 1",
    "#undef HAVE_WCHAR_T": "#define HAVE_WCHAR_T 1",
    "#undef HAVE_X11_KEYSYM_H": "#define HAVE_X11_KEYSYM_H 1",
    "#undef HAVE_X11_XLIB_H": "#define HAVE_X11_XLIB_H 1",
    "#undef HAVE_X11_XUTIL_H": "#define HAVE_X11_XUTIL_H 1",
    "#undef HAVE___PROGNAME": "#define HAVE___PROGNAME 1",
    "#undef HWLOC_C_HAVE_VISIBILITY": "#define HWLOC_C_HAVE_VISIBILITY 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_ALIGNED": "#define HWLOC_HAVE_ATTRIBUTE_ALIGNED 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_ALWAYS_INLINE": "#define HWLOC_HAVE_ATTRIBUTE_ALWAYS_INLINE 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_COLD": "#define HWLOC_HAVE_ATTRIBUTE_COLD 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_CONSTRUCTOR": "#define HWLOC_HAVE_ATTRIBUTE_CONSTRUCTOR 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_CONST": "#define HWLOC_HAVE_ATTRIBUTE_CONST 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_DEPRECATED": "#define HWLOC_HAVE_ATTRIBUTE_DEPRECATED 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_FORMAT": "#define HWLOC_HAVE_ATTRIBUTE_FORMAT 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_HOT": "#define HWLOC_HAVE_ATTRIBUTE_HOT 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_MALLOC": "#define HWLOC_HAVE_ATTRIBUTE_MALLOC 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_MAY_ALIAS": "#define HWLOC_HAVE_ATTRIBUTE_MAY_ALIAS 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_NONNULL": "#define HWLOC_HAVE_ATTRIBUTE_NONNULL 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_NORETURN": "#define HWLOC_HAVE_ATTRIBUTE_NORETURN 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_NO_INSTRUMENT_FUNCTION": "#define HWLOC_HAVE_ATTRIBUTE_NO_INSTRUMENT_FUNCTION 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_PACKED": "#define HWLOC_HAVE_ATTRIBUTE_PACKED 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_PURE": "#define HWLOC_HAVE_ATTRIBUTE_PURE 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_SENTINEL": "#define HWLOC_HAVE_ATTRIBUTE_SENTINEL 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_UNUSED": "#define HWLOC_HAVE_ATTRIBUTE_UNUSED 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_WARN_UNUSED_RESULT": "#define HWLOC_HAVE_ATTRIBUTE_WARN_UNUSED_RESULT 1",
    "#undef HWLOC_HAVE_ATTRIBUTE_WEAK_ALIAS": "#define HWLOC_HAVE_ATTRIBUTE_WEAK_ALIAS 1",
    "#undef HWLOC_HAVE_ATTRIBUTE": "#define HWLOC_HAVE_ATTRIBUTE 1",
    "#undef HWLOC_HAVE_CPU_SET_S": "#define HWLOC_HAVE_CPU_SET_S 1",
    "#undef HWLOC_HAVE_CPU_SET": "#define HWLOC_HAVE_CPU_SET 1",
    "#undef HWLOC_HAVE_DECL_FFSL": "#define HWLOC_HAVE_DECL_FFSL 1",
    "#undef HWLOC_HAVE_DECL_FFS": "#define HWLOC_HAVE_DECL_FFS 1",
    "#undef HWLOC_HAVE_DECL_STRCASECMP": "#define HWLOC_HAVE_DECL_STRCASECMP 1",
    "#undef HWLOC_HAVE_DECL_STRNCASECMP": "#define HWLOC_HAVE_DECL_STRNCASECMP 1",
    "#undef HWLOC_HAVE_FFSL": "#define HWLOC_HAVE_FFSL 1",
    "#undef HWLOC_HAVE_FFS": "#define HWLOC_HAVE_FFS 1",
    "#undef HWLOC_HAVE_LIBTERMCAP": "#define HWLOC_HAVE_LIBTERMCAP 1",
    "#undef HWLOC_HAVE_LINUXIO": "#define HWLOC_HAVE_LINUXIO 1",
    "#undef HWLOC_HAVE_PTHREAD_MUTEX": "#define HWLOC_HAVE_PTHREAD_MUTEX 1",
    "#undef HWLOC_HAVE_SCHED_SETAFFINITY": "#define HWLOC_HAVE_SCHED_SETAFFINITY 1",
    "#undef HWLOC_HAVE_STDINT_H": "#define HWLOC_HAVE_STDINT_H 1",
    "#undef HWLOC_HAVE_SYSCALL": "#define HWLOC_HAVE_SYSCALL 1",
    "#undef HWLOC_HAVE_X11_KEYSYM": "#define HWLOC_HAVE_X11_KEYSYM 1",
    "#undef HWLOC_HAVE_X86_CPUID": "#define HWLOC_HAVE_X86_CPUID 1",
    "#undef HWLOC_SIZEOF_UNSIGNED_INT": "#define HWLOC_SIZEOF_UNSIGNED_INT 4",
    "#undef HWLOC_SIZEOF_UNSIGNED_LONG": "#define HWLOC_SIZEOF_UNSIGNED_LONG 8",
    "#undef HWLOC_SYM_PREFIX_CAPS": "#define HWLOC_SYM_PREFIX_CAPS HWLOC_",
    "#undef HWLOC_SYM_PREFIX": "#define HWLOC_SYM_PREFIX hwloc_",
    "#undef HWLOC_SYM_TRANSFORM": "#define HWLOC_SYM_TRANSFORM 0",
    "#undef HWLOC_USE_NCURSES": "#define HWLOC_USE_NCURSES 1",
    "#undef HWLOC_VERSION_GREEK": "#define HWLOC_VERSION_GREEK \"\"",
    "#undef HWLOC_VERSION_MAJOR": "#define HWLOC_VERSION_MAJOR 2",
    "#undef HWLOC_VERSION_MINOR": "#define HWLOC_VERSION_MINOR 0",
    "#undef HWLOC_VERSION_RELEASE": "#define HWLOC_VERSION_RELEASE 3",
    "#undef HWLOC_VERSION": "#define HWLOC_VERSION \"2.0.3\"",
    "#undef HWLOC_X86_64_ARCH": "#define HWLOC_X86_64_ARCH 1",
    "#undef LT_OBJDIR": "#define LT_OBJDIR \".libs/\"",
    "#undef PACKAGE_BUGREPORT": "#define PACKAGE_BUGREPORT \"http://github.com/open-mpi/hwloc/issues",
    "#undef PACKAGE_NAME": "#define PACKAGE_NAME \"hwloc\"",
    "#undef PACKAGE_STRING": "#define PACKAGE_STRING \"hwloc 2.0.3\"",
    "#undef PACKAGE_TARNAME": "#define PACKAGE_TARNAME \"hwloc\"",
    "#undef PACKAGE_URL": "#define PACKAGE_URL \"\"",
    "#undef PACKAGE_VERSION": "#define PACKAGE_VERSION \"2.0.3\"",
    "#undef PACKAGE": "#define PACKAGE \"hwloc\"",
    "#undef SIZEOF_UNSIGNED_INT": "#define SIZEOF_UNSIGNED_INT 4",
    "#undef SIZEOF_UNSIGNED_LONG": "#define SIZEOF_UNSIGNED_LONG 8",
    "#undef SIZEOF_VOID_P": "#define SIZEOF_VOID_P 8",
    "#undef STDC_HEADERS": "#define STDC_HEADERS 1",
    "# undef _HPUX_SOURCE": "# define _HPUX_SOURCE 1",
    "# undef _ALL_SOURCE": "# define _ALL_SOURCE 1",
    "# undef _GNU_SOURCE": "# define _GNU_SOURCE 1",
    "# undef _POSIX_PTHREAD_SEMANTICS": "# define _POSIX_PTHREAD_SEMANTICS 1",
    "# undef _TANDEM_SOURCE": "# define _TANDEM_SOURCE 1",
    "# undef __EXTENSIONS__": "# define __EXTENSIONS__ 1",
    "#undef VERSION": "#define VERSION \"2.0.3\"",
    "#undef _HPUX_SOURCE": "#define _HPUX_SOURCE 1",
    "#undef hwloc_pid_t": "#define hwloc_pid_t pid_t",
    "#undef hwloc_thread_t": "#define hwloc_thread_t pthread_t",
}

_INCLUDE_PRIVATE_HWLOC_AUTOIGEN_CONFIG_H_CUDA_SUBS = {
    "#undef HAVE_CUDA_RUNTIME_API_H": "#define HAVE_CUDA_RUNTIME_API_H 1",
    "#undef HAVE_CUDA_H": "#define HAVE_CUDA_H 1",
    "#undef HAVE_CUDA": "#define HAVE_CUDA 1",
}

_INCLUDE_PRIVATE_HWLOC_AUTOIGEN_CONFIG_H_LINUX_SUBS = {
    "#undef HAVE_PROGRAM_INVOCATION_NAME": "#define HAVE_PROGRAM_INVOCATION_NAME 1",
    "#undef HWLOC_LINUX_SYS": "#define HWLOC_LINUX_SYS 1",
}

_INCLUDE_PRIVATE_HWLOC_AUTOIGEN_CONFIG_H_LINUX_SUBS.update(_INCLUDE_PRIVATE_HWLOC_AUTOIGEN_CONFIG_H_COMMON_SUBS)

_INCLUDE_PRIVATE_HWLOC_AUTOIGEN_CONFIG_H_CUDA_SUBS.update(_INCLUDE_PRIVATE_HWLOC_AUTOIGEN_CONFIG_H_LINUX_SUBS)

template_rule(
    name = "include_private_hwloc_autogen__config_h",
    src = "include/private/autogen/config.h.in",
    out = "include/private/autogen/config.h",
    substitutions = if_cuda(
        _INCLUDE_PRIVATE_HWLOC_AUTOIGEN_CONFIG_H_CUDA_SUBS,
        if_false = _INCLUDE_PRIVATE_HWLOC_AUTOIGEN_CONFIG_H_LINUX_SUBS,
    ),
)

template_rule(
    name = "move_static_components_h",
    src = "@org_tensorflow//third_party/hwloc:static-components.h",
    out = "hwloc/static-components.h",
    substitutions = {"&hwloc_linuxio_component": "//&hwloc_linuxio_component"},
)

cc_library(
    name = "hwloc",
    srcs = [
        "hwloc/base64.c",
        "hwloc/bind.c",
        "hwloc/bitmap.c",
        "hwloc/components.c",
        "hwloc/cpukinds.c",
        "hwloc/diff.c",
        "hwloc/distances.c",
        "hwloc/memattrs.c",
        "hwloc/misc.c",
        "hwloc/pci-common.c",
        "hwloc/shmem.c",
        "hwloc/static-components.h",
        "hwloc/topology.c",
        "hwloc/topology-hardwired.c",
        "hwloc/topology-noos.c",
        "hwloc/topology-synthetic.c",
        "hwloc/topology-xml.c",
        "hwloc/topology-xml-nolibxml.c",
        "hwloc/traversal.c",
        "include/hwloc/plugins.h",
        "include/hwloc/shmem.h",
        "include/private/autogen/config.h",
        "include/private/components.h",
        "include/private/debug.h",
        "include/private/internal-components.h",
        "include/private/misc.h",
        "include/private/private.h",
        "include/private/xml.h",
    ] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "hwloc/topology-linux.c",
            "include/hwloc/linux.h",
            "hwloc/topology-x86.c",
            "include/private/cpuid-x86.h",
        ],
        "@org_tensorflow//tensorflow:linux_aarch64": [
            "hwloc/topology-linux.c",
            "include/hwloc/linux.h",
        ],
        "@org_tensorflow//tensorflow:linux_ppc64le": [
            "hwloc/topology-linux.c",
            "include/hwloc/linux.h",
        ],
        "@org_tensorflow//tensorflow:freebsd": [
            "hwloc/topology-freebsd.c",
            "hwloc/topology-x86.c",
            "include/private/cpuid-x86.h",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "include/hwloc.h",
        "include/hwloc/autogen/config.h",
        "include/hwloc/bitmap.h",
        "include/hwloc/cpukinds.h",
        "include/hwloc/deprecated.h",
        "include/hwloc/diff.h",
        "include/hwloc/distances.h",
        "include/hwloc/export.h",
        "include/hwloc/helper.h",
        "include/hwloc/inlines.h",
        "include/hwloc/memattrs.h",
        "include/hwloc/rename.h",
    ],
    copts = COMMON_INCLUDE_COPTS + DISABLE_WARNINGS_COPTS + VAR_SETTINGS_COPTS,
    features = [
        "-parse_headers",
        "-layering_check",
    ],
    includes = [
        "hwloc",
        "include",
    ],
    deps = [],
)

cc_binary(
    name = "hwloc_print",
    srcs = ["hwloc_print.cc"],
    copts = COMMON_INCLUDE_COPTS,
    deps = [
        ":hwloc",
    ],
)
