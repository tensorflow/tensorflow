# Description:
# jemalloc - a general-purpose scalable concurrent malloc implementation

licenses(["notice"])  # BSD

exports_files(["COPYING"])

load("@%ws%//third_party:common.bzl", "template_rule")

cc_library(
    name = "jemalloc",
    srcs = [
        "src/arena.c",
        "src/atomic.c",
        "src/base.c",
        "src/bitmap.c",
        "src/chunk.c",
        "src/chunk_dss.c",
        "src/chunk_mmap.c",
        "src/ckh.c",
        "src/ctl.c",
        "src/extent.c",
        "src/hash.c",
        "src/huge.c",
        "src/jemalloc.c",
        "src/mb.c",
        "src/mutex.c",
        "src/nstime.c",
        "src/pages.c",
        "src/prng.c",
        "src/prof.c",
        "src/quarantine.c",
        "src/rtree.c",
        "src/spin.c",
        "src/stats.c",
        "src/tcache.c",
        "src/tsd.c",
        "src/util.c",
        "src/witness.c",
    ],
    hdrs = [
        "include/jemalloc/internal/arena.h",
        "include/jemalloc/internal/assert.h",
        "include/jemalloc/internal/atomic.h",
        "include/jemalloc/internal/base.h",
        "include/jemalloc/internal/bitmap.h",
        "include/jemalloc/internal/chunk.h",
        "include/jemalloc/internal/chunk_dss.h",
        "include/jemalloc/internal/chunk_mmap.h",
        "include/jemalloc/internal/ckh.h",
        "include/jemalloc/internal/ctl.h",
        "include/jemalloc/internal/extent.h",
        "include/jemalloc/internal/hash.h",
        "include/jemalloc/internal/huge.h",
        "include/jemalloc/internal/jemalloc_internal.h",
        "include/jemalloc/internal/jemalloc_internal_decls.h",
        "include/jemalloc/internal/jemalloc_internal_defs.h",
        "include/jemalloc/internal/jemalloc_internal_macros.h",
        "include/jemalloc/internal/mb.h",
        "include/jemalloc/internal/mutex.h",
        "include/jemalloc/internal/nstime.h",
        "include/jemalloc/internal/pages.h",
        "include/jemalloc/internal/ph.h",
        "include/jemalloc/internal/private_namespace.h",
        "include/jemalloc/internal/prng.h",
        "include/jemalloc/internal/prof.h",
        "include/jemalloc/internal/ql.h",
        "include/jemalloc/internal/qr.h",
        "include/jemalloc/internal/quarantine.h",
        "include/jemalloc/internal/rb.h",
        "include/jemalloc/internal/rtree.h",
        "include/jemalloc/internal/size_classes.h",
        "include/jemalloc/internal/smoothstep.h",
        "include/jemalloc/internal/spin.h",
        "include/jemalloc/internal/stats.h",
        "include/jemalloc/internal/tcache.h",
        "include/jemalloc/internal/ticker.h",
        "include/jemalloc/internal/tsd.h",
        "include/jemalloc/internal/util.h",
        "include/jemalloc/internal/valgrind.h",
        "include/jemalloc/internal/witness.h",
        "include/jemalloc/jemalloc.h",
    ],
    # Same flags that jemalloc uses to build.
    copts = [
        "-O3",
        "-funroll-loops",
        "-D_GNU_SOURCE",
        "-D_REENTRANT",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "jemalloc_sh",
    srcs = ["include/jemalloc/jemalloc.sh"],
)

genrule(
    name = "jemalloc_h",
    srcs = [
        ":jemalloc_defs_h",
        ":jemalloc_macros_h",
        ":jemalloc_mangle_h",
        ":jemalloc_protos_h",
        ":jemalloc_rename_h",
        ":jemalloc_typedefs_h",
    ],
    outs = ["include/jemalloc/jemalloc.h"],
    cmd = "$(location :jemalloc_sh) $$(dirname $(location :jemalloc_defs_h))/../../ >$@",
    tools = [":jemalloc_sh"],
)

# Add to this list if you want to export more symbols from jemalloc.
genrule(
    name = "public_symbols_txt",
    outs = ["include/jemalloc/internal/public_symbols.txt"],
    cmd = "\n".join([
        "cat <<'EOF' > $@",
        "free:jemalloc_free",
        "malloc:jemalloc_malloc",
        "posix_memalign:jemalloc_posix_memalign",
        "realloc:jemalloc_realloc",
        "EOF",
    ]),
)

sh_binary(
    name = "jemalloc_mangle_sh",
    srcs = ["include/jemalloc/jemalloc_mangle.sh"],
)

genrule(
    name = "jemalloc_mangle_h",
    srcs = [":public_symbols_txt"],
    outs = ["include/jemalloc/jemalloc_mangle.h"],
    cmd = "$(location :jemalloc_mangle_sh) $(location :public_symbols_txt) je_ >$@",
    tools = [":jemalloc_mangle_sh"],
)

sh_binary(
    name = "jemalloc_rename_sh",
    srcs = ["include/jemalloc/jemalloc_rename.sh"],
)

genrule(
    name = "jemalloc_rename_h",
    srcs = [":public_symbols_txt"],
    outs = ["include/jemalloc/jemalloc_rename.h"],
    cmd = "$(location :jemalloc_rename_sh) $(location :public_symbols_txt) >$@",
    tools = [":jemalloc_rename_sh"],
)

sh_binary(
    name = "private_namespace_sh",
    srcs = ["include/jemalloc/internal/private_namespace.sh"],
)

genrule(
    name = "private_namespace_h",
    srcs = ["include/jemalloc/internal/private_symbols.txt"],
    outs = ["include/jemalloc/internal/private_namespace.h"],
    cmd = "$(location :private_namespace_sh) $(location include/jemalloc/internal/private_symbols.txt) >$@",
    tools = [":private_namespace_sh"],
)

sh_binary(
    name = "public_namespace_sh",
    srcs = ["include/jemalloc/internal/public_namespace.sh"],
)

genrule(
    name = "public_namespace_h",
    srcs = [":public_symbols_txt"],
    outs = ["include/jemalloc/internal/public_namespace.h"],
    cmd = "$(location :public_namespace_sh) $(location :public_symbols_txt) >$@",
    tools = [":public_namespace_sh"],
)

sh_binary(
    name = "size_classes_sh",
    srcs = ["include/jemalloc/internal/size_classes.sh"],
)

# Size classes for Linux x86_64. Update if adding builds for other
# architectures. See size_classes.sh for details on the arguments.
genrule(
    name = "size_classes_h",
    outs = ["include/jemalloc/internal/size_classes.h"],
    cmd = "$(location :size_classes_sh) \"3 4\" 3 12 2 >$@",
    tools = [":size_classes_sh"],
)

template_rule(
    name = "jemalloc_internal_h",
    src = "include/jemalloc/internal/jemalloc_internal.h.in",
    out = "include/jemalloc/internal/jemalloc_internal.h",
    substitutions = {
        "@private_namespace@": "je_",
        "@install_suffix@": "",
    },
)

template_rule(
    name = "jemalloc_internal_defs_h",
    src = "include/jemalloc/internal/jemalloc_internal_defs.h.in",
    out = "include/jemalloc/internal/jemalloc_internal_defs.h",
    substitutions = {
        "#undef JEMALLOC_PREFIX": "#define JEMALLOC_PREFIX \"jemalloc_\"",
        "#undef JEMALLOC_CPREFIX": "#define JEMALLOC_CPREFIX \"JEMALLOC_\"",
        "#undef JEMALLOC_PRIVATE_NAMESPACE": "#define JEMALLOC_PRIVATE_NAMESPACE je_",
        "#undef CPU_SPINWAIT": "#define CPU_SPINWAIT __asm__ volatile(\"pause\")",
        "#undef JEMALLOC_HAVE_BUILTIN_CLZ": "#define JEMALLOC_HAVE_BUILTIN_CLZ",
        "#undef JEMALLOC_USE_SYSCALL": "#define JEMALLOC_USE_SYSCALL",
        "#undef JEMALLOC_HAVE_SECURE_GETENV": "#define JEMALLOC_HAVE_SECURE_GETENV",
        "#undef JEMALLOC_HAVE_PTHREAD_ATFORK": "#define JEMALLOC_HAVE_PTHREAD_ATFORK",
        "#undef JEMALLOC_HAVE_CLOCK_MONOTONIC_COARSE": "#define JEMALLOC_HAVE_CLOCK_MONOTONIC_COARSE 1",
        # Newline required because of substitution conflicts.
        "#undef JEMALLOC_HAVE_CLOCK_MONOTONIC\n": "#define JEMALLOC_HAVE_CLOCK_MONOTONIC 1\n",
        "#undef JEMALLOC_THREADED_INIT": "#define JEMALLOC_THREADED_INIT",
        "#undef JEMALLOC_TLS_MODEL": "#define JEMALLOC_TLS_MODEL __attribute__((tls_model(\"initial-exec\")))",
        "#undef JEMALLOC_CC_SILENCE": "#define JEMALLOC_CC_SILENCE",
        "#undef JEMALLOC_STATS": "#define JEMALLOC_STATS",
        "#undef JEMALLOC_TCACHE": "#define JEMALLOC_TCACHE",
        "#undef JEMALLOC_DSS": "#define JEMALLOC_DSS",
        "#undef JEMALLOC_FILL": "#define JEMALLOC_FILL",
        "#undef LG_TINY_MIN": "#define LG_TINY_MIN 3",
        "#undef LG_PAGE": "#define LG_PAGE 12",
        "#undef JEMALLOC_MAPS_COALESCE": "#define JEMALLOC_MAPS_COALESCE",
        "#undef JEMALLOC_TLS": "#define JEMALLOC_TLS",
        "#undef JEMALLOC_INTERNAL_UNREACHABLE": "#define JEMALLOC_INTERNAL_UNREACHABLE __builtin_unreachable",
        "#undef JEMALLOC_INTERNAL_FFSLL": "#define JEMALLOC_INTERNAL_FFSLL __builtin_ffsll",
        # Newline required because of substitution conflicts.
        "#undef JEMALLOC_INTERNAL_FFSL\n": "#define JEMALLOC_INTERNAL_FFSL __builtin_ffsl\n",
        "#undef JEMALLOC_INTERNAL_FFS\n": "#define JEMALLOC_INTERNAL_FFS __builtin_ffs\n",
        "#undef JEMALLOC_CACHE_OBLIVIOUS": "#define JEMALLOC_CACHE_OBLIVIOUS",
        "#undef JEMALLOC_PROC_SYS_VM_OVERCOMMIT_MEMORY": "#define JEMALLOC_PROC_SYS_VM_OVERCOMMIT_MEMORY",
        "#undef JEMALLOC_HAVE_MADVISE": "#define JEMALLOC_HAVE_MADVISE",
        "#undef JEMALLOC_PURGE_MADVISE_DONTNEED": "#define JEMALLOC_PURGE_MADVISE_DONTNEED",
        "#undef JEMALLOC_THP": "#define JEMALLOC_THP",
        "#undef JEMALLOC_HAS_ALLOCA_H": "#define JEMALLOC_HAS_ALLOCA_H 1",
        # Newline required because of substitution conflicts.
        "#undef LG_SIZEOF_INT\n": "#define LG_SIZEOF_INT 2\n",
        "#undef LG_SIZEOF_LONG\n": "#define LG_SIZEOF_LONG 3\n",
        "#undef LG_SIZEOF_LONG_LONG": "#define LG_SIZEOF_LONG_LONG 3",
        "#undef LG_SIZEOF_INTMAX_T": "#define LG_SIZEOF_INTMAX_T 3",
        "#undef JEMALLOC_GLIBC_MALLOC_HOOK": "#define JEMALLOC_GLIBC_MALLOC_HOOK",
        "#undef JEMALLOC_GLIBC_MEMALIGN_HOOK": "#define JEMALLOC_GLIBC_MEMALIGN_HOOK",
        "#undef JEMALLOC_HAVE_PTHREAD_MUTEX_ADAPTIVE_NP": "#define JEMALLOC_HAVE_PTHREAD_MUTEX_ADAPTIVE_NP",
        "#undef JEMALLOC_CONFIG_MALLOC_CONF": "#define JEMALLOC_CONFIG_MALLOC_CONF \"\"",
    },
)

template_rule(
    name = "jemalloc_defs_h",
    src = "include/jemalloc/jemalloc_defs.h.in",
    out = "include/jemalloc/jemalloc_defs.h",
    substitutions = {
        "#undef JEMALLOC_HAVE_ATTR": "#define JEMALLOC_HAVE_ATTR",
        "#undef JEMALLOC_HAVE_ATTR_ALLOC_SIZE": "#define JEMALLOC_HAVE_ATTR_ALLOC_SIZE",
        "#undef JEMALLOC_HAVE_ATTR_FORMAT_GNU_PRINTF": "#define JEMALLOC_HAVE_ATTR_FORMAT_GNU_PRINTF",
        "#undef JEMALLOC_HAVE_ATTR_FORMAT_PRINTF": "#define JEMALLOC_HAVE_ATTR_FORMAT_PRINTF",
        "#undef JEMALLOC_OVERRIDE_MEMALIGN": "#define JEMALLOC_OVERRIDE_MEMALIGN",
        "#undef JEMALLOC_OVERRIDE_VALLOC": "#define JEMALLOC_OVERRIDE_VALLOC",
        "#undef JEMALLOC_USABLE_SIZE_CONST": "#define JEMALLOC_USABLE_SIZE_CONST",
        "#undef JEMALLOC_USE_CXX_THROW": "#define JEMALLOC_USE_CXX_THROW",
        "#undef LG_SIZEOF_PTR": "#define LG_SIZEOF_PTR 3",
    },
)

template_rule(
    name = "jemalloc_macros_h",
    src = "include/jemalloc/jemalloc_macros.h.in",
    out = "include/jemalloc/jemalloc_macros.h",
    substitutions = {
        "@jemalloc_version@": "0.0.0",
        "@jemalloc_version_major@": "0",
        "@jemalloc_version_minor@": "0",
        "@jemalloc_version_bugfix@": "0",
        "@jemalloc_version_nrev@": "0",
        "@jemalloc_version_gid@": "0000000000000000000000000000000000000000",
    },
)

template_rule(
    name = "jemalloc_protos_h",
    src = "include/jemalloc/jemalloc_protos.h.in",
    out = "include/jemalloc/jemalloc_protos.h",
    substitutions = {
        "@aligned_alloc": "aligned_alloc",
        "@calloc": "calloc",
        "@cbopaque": "cbopaque",
        "@dallocx": "dallocx",
        "@free": "free",
        "@je": "je",
        "@mallctl": "mallctl",
        "@mallctlnametomib": "mallctlnametomib",
        "@mallctlbymib": "mallctlbymib",
        "@malloc_stats_print": "malloc_stats_print",
        "@malloc_usable_size": "malloc_usable_size",
        "@malloc": "malloc",
        "@mallocx": "mallocx",
        "@memalign": "memalign",
        "@nallocx": "nallocx",
        "@posix_memalign": "posix_memalign",
        "@rallocx": "rallocx",
        "@realloc": "realloc",
        "@sallocx": "sallocx",
        "@sdallocx": "sdallocx",
        "@valloc": "valloc",
        "@xallocx": "xallocx",
    },
)

template_rule(
    name = "jemalloc_typedefs_h",
    src = "include/jemalloc/jemalloc_typedefs.h.in",
    out = "include/jemalloc/jemalloc_typedefs.h",
    substitutions = {},
)
