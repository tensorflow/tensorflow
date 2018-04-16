# Bazel BUILD file for LLVM.
#
# This BUILD file is auto-generated; do not edit!

licenses(["notice"])

exports_files(["LICENSE.TXT"])

load(
    "@org_tensorflow//third_party/llvm:llvm.bzl",
    "gentbl",
    "expand_cmake_vars",
    "llvm_target_cmake_vars",
    "cmake_var_string",
)
load(
    "@org_tensorflow//third_party:common.bzl",
    "template_rule",
)

package(default_visibility = ["//visibility:public"])

llvm_host_triple = "x86_64-unknown-linux_gnu"

llvm_targets = [
    "AArch64",
    # Uncomment to enable the AMDGPU backend.
    # TODO(phawkins): use a configure-time test.
    # "AMDGPU",
    "ARM",
    "NVPTX",
    "PowerPC",
    "X86",
]

llvm_target_asm_parsers = llvm_targets

llvm_target_asm_printers = llvm_targets

llvm_target_disassemblers = llvm_targets

# TODO(phawkins): the set of CMake variables was hardcoded for expediency.
# However, we should really detect many of these via configure-time tests.

# The set of CMake variables common to all targets.
cmake_vars = {
    # Headers
    "HAVE_DIRENT_H": 1,
    "HAVE_DLFCN_H": 1,
    "HAVE_ERRNO_H": 1,
    "HAVE_EXECINFO_H": 1,
    "HAVE_FCNTL_H": 1,
    "HAVE_INTTYPES_H": 1,
    "HAVE_PTHREAD_H": 1,
    "HAVE_SIGNAL_H": 1,
    "HAVE_STDINT_H": 1,
    "HAVE_SYS_IOCTL_H": 1,
    "HAVE_SYS_MMAN_H": 1,
    "HAVE_SYS_PARAM_H": 1,
    "HAVE_SYS_RESOURCE_H": 1,
    "HAVE_SYS_STAT_H": 1,
    "HAVE_SYS_TIME_H": 1,
    "HAVE_SYS_TYPES_H": 1,
    "HAVE_TERMIOS_H": 1,
    "HAVE_UNISTD_H": 1,
    "HAVE_ZLIB_H": 1,

    # Features
    "HAVE_BACKTRACE": 1,
    "BACKTRACE_HEADER": "execinfo.h",
    "HAVE_DLOPEN": 1,
    "HAVE_FUTIMES": 1,
    "HAVE_GETCWD": 1,
    "HAVE_GETPAGESIZE": 1,
    "HAVE_GETRLIMIT": 1,
    "HAVE_GETRUSAGE": 1,
    "HAVE_GETTIMEOFDAY": 1,
    "HAVE_INT64_T": 1,
    "HAVE_ISATTY": 1,
    "HAVE_LIBEDIT": 1,
    "HAVE_LIBPTHREAD": 1,
    "HAVE_LIBZ": 1,
    "HAVE_MKDTEMP": 1,
    "HAVE_MKSTEMP": 1,
    "HAVE_MKTEMP": 1,
    "HAVE_PREAD": 1,
    "HAVE_PTHREAD_GETSPECIFIC": 1,
    "HAVE_PTHREAD_MUTEX_LOCK": 1,
    "HAVE_PTHREAD_RWLOCK_INIT": 1,
    "HAVE_REALPATH": 1,
    "HAVE_SBRK": 1,
    "HAVE_SETENV": 1,
    "HAVE_SETRLIMIT": 1,
    "HAVE_SIGALTSTACK": 1,
    "HAVE_STRERROR": 1,
    "HAVE_STRERROR_R": 1,
    "HAVE_STRTOLL": 1,
    "HAVE_SYSCONF": 1,
    "HAVE_UINT64_T": 1,
    "HAVE__UNWIND_BACKTRACE": 1,

    # LLVM features
    "ENABLE_BACKTRACES": 1,
    "LLVM_BINDIR": "/dev/null",
    "LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING": 0,
    "LLVM_ENABLE_ABI_BREAKING_CHECKS": 0,
    "LLVM_ENABLE_THREADS": 1,
    "LLVM_ENABLE_ZLIB": 1,
    "LLVM_HAS_ATOMICS": 1,
    "LLVM_INCLUDEDIR": "/dev/null",
    "LLVM_INFODIR": "/dev/null",
    "LLVM_MANDIR": "/dev/null",
    "LLVM_NATIVE_TARGET": 1,
    "LLVM_NATIVE_TARGETINFO": 1,
    "LLVM_NATIVE_TARGETMC": 1,
    "LLVM_NATIVE_ASMPRINTER": 1,
    "LLVM_NATIVE_ASMPARSER": 1,
    "LLVM_NATIVE_DISASSEMBLER": 1,
    "LLVM_ON_UNIX": 1,
    "LLVM_PREFIX": "/dev/null",
    "LLVM_VERSION_MAJOR": 0,
    "LLVM_VERSION_MINOR": 0,
    "LLVM_VERSION_PATCH": 0,
    "LTDL_SHLIB_EXT": ".so",
    "PACKAGE_NAME": "llvm",
    "PACKAGE_STRING": "llvm tensorflow-trunk",
    "PACKAGE_VERSION": "tensorflow-trunk",
    "RETSIGTYPE": "void",
}

# CMake variables specific to the Linux platform
linux_cmake_vars = {
    "HAVE_MALLOC_H": 1,
    "HAVE_LINK_H": 1,
    "HAVE_MALLINFO": 1,
    "HAVE_FUTIMENS": 1,
}

# CMake variables specific to the Darwin (Mac OS X) platform.
darwin_cmake_vars = {
    "HAVE_MALLOC_MALLOC_H": 1,
}

# Select a set of CMake variables based on the platform.
# TODO(phawkins): use a better method to select the right host triple, rather
# than hardcoding x86_64.
all_cmake_vars = select({
    "@org_tensorflow//tensorflow:darwin": cmake_var_string(
        cmake_vars + llvm_target_cmake_vars("X86", "x86_64-apple-darwin") +
        darwin_cmake_vars,
    ),
    "@org_tensorflow//tensorflow:linux_ppc64le": cmake_var_string(
        cmake_vars +
        llvm_target_cmake_vars("PowerPC", "powerpc64le-unknown-linux_gnu") +
        linux_cmake_vars,
    ),
    "//conditions:default": cmake_var_string(
        cmake_vars +
        llvm_target_cmake_vars("X86", "x86_64-unknown-linux_gnu") +
        linux_cmake_vars,
    ),
})

# Performs CMake variable substitutions on configuration header files.
expand_cmake_vars(
    name = "config_gen",
    src = "include/llvm/Config/config.h.cmake",
    cmake_vars = all_cmake_vars,
    dst = "include/llvm/Config/config.h",
)

expand_cmake_vars(
    name = "llvm_config_gen",
    src = "include/llvm/Config/llvm-config.h.cmake",
    cmake_vars = all_cmake_vars,
    dst = "include/llvm/Config/llvm-config.h",
)

expand_cmake_vars(
    name = "abi_breaking_gen",
    src = "include/llvm/Config/abi-breaking.h.cmake",
    cmake_vars = all_cmake_vars,
    dst = "include/llvm/Config/abi-breaking.h",
)

# Performs macro expansions on .def.in files
template_rule(
    name = "targets_def_gen",
    src = "include/llvm/Config/Targets.def.in",
    out = "include/llvm/Config/Targets.def",
    substitutions = {
        "@LLVM_ENUM_TARGETS@": "\n".join(
            ["LLVM_TARGET({})".format(t) for t in llvm_targets],
        ),
    },
)

template_rule(
    name = "asm_parsers_def_gen",
    src = "include/llvm/Config/AsmParsers.def.in",
    out = "include/llvm/Config/AsmParsers.def",
    substitutions = {
        "@LLVM_ENUM_ASM_PARSERS@": "\n".join(
            ["LLVM_ASM_PARSER({})".format(t) for t in llvm_target_asm_parsers],
        ),
    },
)

template_rule(
    name = "asm_printers_def_gen",
    src = "include/llvm/Config/AsmPrinters.def.in",
    out = "include/llvm/Config/AsmPrinters.def",
    substitutions = {
        "@LLVM_ENUM_ASM_PRINTERS@": "\n".join(
            ["LLVM_ASM_PRINTER({})".format(t) for t in llvm_target_asm_printers],
        ),
    },
)

template_rule(
    name = "disassemblers_def_gen",
    src = "include/llvm/Config/Disassemblers.def.in",
    out = "include/llvm/Config/Disassemblers.def",
    substitutions = {
        "@LLVM_ENUM_DISASSEMBLERS@": "\n".join(
            ["LLVM_DISASSEMBLER({})".format(t) for t in llvm_target_disassemblers],
        ),
    },
)

# A common library that all LLVM targets depend on.
cc_library(
    name = "config",
    hdrs = [
        "include/llvm/Config/AsmParsers.def",
        "include/llvm/Config/AsmPrinters.def",
        "include/llvm/Config/Disassemblers.def",
        "include/llvm/Config/Targets.def",
        "include/llvm/Config/abi-breaking.h",
        "include/llvm/Config/config.h",
        "include/llvm/Config/llvm-config.h",
    ],
    defines = [
        "LLVM_ENABLE_STATS",
        "__STDC_LIMIT_MACROS",
        "__STDC_CONSTANT_MACROS",
        "__STDC_FORMAT_MACROS",
        "_DEBUG",
        "LLVM_BUILD_GLOBAL_ISEL",
    ],
    includes = ["include"],
)

# A creator of an empty file include/llvm/Support/VCSRevision.h.
# This is usually populated by the upstream build infrastructure, but in this
# case we leave it blank. See upstream revision r300160.
genrule(
    name = "vcs_revision_gen",
    srcs = [],
    outs = ["include/llvm/Support/VCSRevision.h"],
    cmd = "echo '' > \"$@\"",
)

# Rules that apply the LLVM tblgen tool.
gentbl(
    name = "intrinsics_gen",
    tbl_outs = [("-gen-intrinsic", "include/llvm/IR/Intrinsics.gen")],
    tblgen = ":llvm-tblgen",
    td_file = "include/llvm/IR/Intrinsics.td",
    td_srcs = glob([
        "include/llvm/CodeGen/*.td",
        "include/llvm/IR/Intrinsics*.td",
    ]),
)

gentbl(
    name = "attributes_gen",
    tbl_outs = [("-gen-attrs", "include/llvm/IR/Attributes.gen")],
    tblgen = ":llvm-tblgen",
    td_file = "include/llvm/IR/Attributes.td",
    td_srcs = ["include/llvm/IR/Attributes.td"],
)

gentbl(
    name = "attributes_compat_gen",
    tbl_outs = [("-gen-attrs", "lib/IR/AttributesCompatFunc.inc")],
    tblgen = ":llvm-tblgen",
    td_file = "lib/IR/AttributesCompatFunc.td",
    td_srcs = [
        "lib/IR/AttributesCompatFunc.td",
        "include/llvm/IR/Attributes.td",
    ],
)

# Binary targets used by Tensorflow.
cc_binary(
    name = "llvm-tblgen",
    srcs = glob([
        "utils/TableGen/*.cpp",
        "utils/TableGen/*.h",
    ]),
    linkopts = [
        "-lm",
        "-ldl",
        "-lpthread",
    ],
    stamp = 0,
    deps = [
        ":config",
        ":support",
        ":table_gen",
    ],
)

cc_binary(
    name = "FileCheck",
    testonly = 1,
    srcs = glob([
        "utils/FileCheck/*.cpp",
        "utils/FileCheck/*.h",
    ]),
    linkopts = [
        "-ldl",
        "-lm",
        "-lpthread",
    ],
    stamp = 0,
    deps = [":support"],
)

llvm_target_list = [
    {
        "name": "AArch64",
        "lower_name": "aarch64",
        "short_name": "AArch64",
        "tbl_outs": [
            ("-gen-register-bank", "lib/Target/AArch64/AArch64GenRegisterBank.inc"),
            ("-gen-register-info", "lib/Target/AArch64/AArch64GenRegisterInfo.inc"),
            ("-gen-instr-info", "lib/Target/AArch64/AArch64GenInstrInfo.inc"),
            ("-gen-emitter", "lib/Target/AArch64/AArch64GenMCCodeEmitter.inc"),
            ("-gen-pseudo-lowering", "lib/Target/AArch64/AArch64GenMCPseudoLowering.inc"),
            ("-gen-asm-writer", "lib/Target/AArch64/AArch64GenAsmWriter.inc"),
            ("-gen-asm-writer -asmwriternum=1", "lib/Target/AArch64/AArch64GenAsmWriter1.inc"),
            ("-gen-asm-matcher", "lib/Target/AArch64/AArch64GenAsmMatcher.inc"),
            ("-gen-dag-isel", "lib/Target/AArch64/AArch64GenDAGISel.inc"),
            ("-gen-fast-isel", "lib/Target/AArch64/AArch64GenFastISel.inc"),
            ("-gen-global-isel", "lib/Target/AArch64/AArch64GenGlobalISel.inc"),
            ("-gen-callingconv", "lib/Target/AArch64/AArch64GenCallingConv.inc"),
            ("-gen-subtarget", "lib/Target/AArch64/AArch64GenSubtargetInfo.inc"),
            ("-gen-disassembler", "lib/Target/AArch64/AArch64GenDisassemblerTables.inc"),
            ("-gen-searchable-tables", "lib/Target/AArch64/AArch64GenSystemOperands.inc"),
        ],
    },
    {
        "name": "AMDGPU",
        "lower_name": "amdgpu",
        "short_name": "AMDGPU",
        "tbl_outs": [
            ("-gen-register-bank", "lib/Target/AMDGPU/AMDGPUGenRegisterBank.inc"),
            ("-gen-register-info", "lib/Target/AMDGPU/AMDGPUGenRegisterInfo.inc"),
            ("-gen-instr-info", "lib/Target/AMDGPU/AMDGPUGenInstrInfo.inc"),
            ("-gen-dag-isel", "lib/Target/AMDGPU/AMDGPUGenDAGISel.inc"),
            ("-gen-callingconv", "lib/Target/AMDGPU/AMDGPUGenCallingConv.inc"),
            ("-gen-subtarget", "lib/Target/AMDGPU/AMDGPUGenSubtargetInfo.inc"),
            ("-gen-tgt-intrinsic", "lib/Target/AMDGPU/AMDGPUGenIntrinsics.inc"),
            ("-gen-emitter", "lib/Target/AMDGPU/AMDGPUGenMCCodeEmitter.inc"),
            ("-gen-dfa-packetizer", "lib/Target/AMDGPU/AMDGPUGenDFAPacketizer.inc"),
            ("-gen-asm-writer", "lib/Target/AMDGPU/AMDGPUGenAsmWriter.inc"),
            ("-gen-asm-matcher", "lib/Target/AMDGPU/AMDGPUGenAsmMatcher.inc"),
            ("-gen-disassembler", "lib/Target/AMDGPU/AMDGPUGenDisassemblerTables.inc"),
            ("-gen-pseudo-lowering", "lib/Target/AMDGPU/AMDGPUGenMCPseudoLowering.inc"),
        ],
    },
    {
        "name": "ARM",
        "lower_name": "arm",
        "short_name": "ARM",
        "tbl_outs": [
            ("-gen-register-bank", "lib/Target/ARM/ARMGenRegisterBank.inc"),
            ("-gen-register-info", "lib/Target/ARM/ARMGenRegisterInfo.inc"),
            ("-gen-searchable-tables", "lib/Target/ARM/ARMGenSystemRegister.inc"),
            ("-gen-instr-info", "lib/Target/ARM/ARMGenInstrInfo.inc"),
            ("-gen-emitter", "lib/Target/ARM/ARMGenMCCodeEmitter.inc"),
            ("-gen-pseudo-lowering", "lib/Target/ARM/ARMGenMCPseudoLowering.inc"),
            ("-gen-asm-writer", "lib/Target/ARM/ARMGenAsmWriter.inc"),
            ("-gen-asm-matcher", "lib/Target/ARM/ARMGenAsmMatcher.inc"),
            ("-gen-dag-isel", "lib/Target/ARM/ARMGenDAGISel.inc"),
            ("-gen-fast-isel", "lib/Target/ARM/ARMGenFastISel.inc"),
            ("-gen-global-isel", "lib/Target/ARM/ARMGenGlobalISel.inc"),
            ("-gen-callingconv", "lib/Target/ARM/ARMGenCallingConv.inc"),
            ("-gen-subtarget", "lib/Target/ARM/ARMGenSubtargetInfo.inc"),
            ("-gen-disassembler", "lib/Target/ARM/ARMGenDisassemblerTables.inc"),
        ],
    },
    {
        "name": "NVPTX",
        "lower_name": "nvptx",
        "short_name": "NVPTX",
        "tbl_outs": [
            ("-gen-register-info", "lib/Target/NVPTX/NVPTXGenRegisterInfo.inc"),
            ("-gen-instr-info", "lib/Target/NVPTX/NVPTXGenInstrInfo.inc"),
            ("-gen-asm-writer", "lib/Target/NVPTX/NVPTXGenAsmWriter.inc"),
            ("-gen-dag-isel", "lib/Target/NVPTX/NVPTXGenDAGISel.inc"),
            ("-gen-subtarget", "lib/Target/NVPTX/NVPTXGenSubtargetInfo.inc"),
        ],
    },
    {
        "name": "PowerPC",
        "lower_name": "powerpc",
        "short_name": "PPC",
        "tbl_outs": [
            ("-gen-asm-writer", "lib/Target/PowerPC/PPCGenAsmWriter.inc"),
            ("-gen-asm-matcher", "lib/Target/PowerPC/PPCGenAsmMatcher.inc"),
            ("-gen-emitter", "lib/Target/PowerPC/PPCGenMCCodeEmitter.inc"),
            ("-gen-register-info", "lib/Target/PowerPC/PPCGenRegisterInfo.inc"),
            ("-gen-instr-info", "lib/Target/PowerPC/PPCGenInstrInfo.inc"),
            ("-gen-dag-isel", "lib/Target/PowerPC/PPCGenDAGISel.inc"),
            ("-gen-fast-isel", "lib/Target/PowerPC/PPCGenFastISel.inc"),
            ("-gen-callingconv", "lib/Target/PowerPC/PPCGenCallingConv.inc"),
            ("-gen-subtarget", "lib/Target/PowerPC/PPCGenSubtargetInfo.inc"),
            ("-gen-disassembler", "lib/Target/PowerPC/PPCGenDisassemblerTables.inc"),
        ],
    },
    {
        "name": "X86",
        "lower_name": "x86",
        "short_name": "X86",
        "tbl_outs": [
            ("-gen-register-bank", "lib/Target/X86/X86GenRegisterBank.inc"),
            ("-gen-register-info", "lib/Target/X86/X86GenRegisterInfo.inc"),
            ("-gen-disassembler", "lib/Target/X86/X86GenDisassemblerTables.inc"),
            ("-gen-instr-info", "lib/Target/X86/X86GenInstrInfo.inc"),
            ("-gen-asm-writer", "lib/Target/X86/X86GenAsmWriter.inc"),
            ("-gen-asm-writer -asmwriternum=1", "lib/Target/X86/X86GenAsmWriter1.inc"),
            ("-gen-asm-matcher", "lib/Target/X86/X86GenAsmMatcher.inc"),
            ("-gen-dag-isel", "lib/Target/X86/X86GenDAGISel.inc"),
            ("-gen-fast-isel", "lib/Target/X86/X86GenFastISel.inc"),
            ("-gen-global-isel", "lib/Target/X86/X86GenGlobalISel.inc"),
            ("-gen-callingconv", "lib/Target/X86/X86GenCallingConv.inc"),
            ("-gen-subtarget", "lib/Target/X86/X86GenSubtargetInfo.inc"),
            ("-gen-x86-EVEX2VEX-tables", "lib/Target/X86/X86GenEVEX2VEXTables.inc"),
        ],
    },
]

[
    gentbl(
        name = target["lower_name"] + "_target_gen",
        tbl_outs = target["tbl_outs"],
        tblgen = ":llvm-tblgen",
        td_file = ("lib/Target/" + target["name"] + "/" + target["short_name"] +
                   ".td"),
        td_srcs = glob([
            "lib/Target/" + target["name"] + "/*.td",
            "include/llvm/CodeGen/*.td",
            "include/llvm/IR/Intrinsics*.td",
            "include/llvm/TableGen/*.td",
            "include/llvm/Target/*.td",
            "include/llvm/Target/GlobalISel/*.td",
        ]),
    )
    for target in llvm_target_list
]

# This target is used to provide *.def files to x86_code_gen.
# Files with '.def' extension are not allowed in 'srcs' of 'cc_library' rule.
cc_library(
    name = "x86_defs",
    hdrs = glob([
        "lib/Target/X86/*.def",
    ]),
    visibility = ["//visibility:private"],
)

# This filegroup provides the docker build script in LLVM repo
filegroup(
    name = "docker",
    srcs = glob([
        "utils/docker/build_docker_image.sh",
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "aarch64_asm_parser",
    srcs = glob([
        "lib/Target/AArch64/AsmParser/*.c",
        "lib/Target/AArch64/AsmParser/*.cpp",
        "lib/Target/AArch64/AsmParser/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AArch64/AsmParser/*.h",
        "include/llvm/Target/AArch64/AsmParser/*.def",
        "include/llvm/Target/AArch64/AsmParser/*.inc",
        "lib/Target/AArch64/AsmParser/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AArch64"],
    deps = [
        ":aarch64_desc",
        ":aarch64_info",
        ":aarch64_utils",
        ":config",
        ":mc",
        ":mc_parser",
        ":support",
    ],
)

cc_library(
    name = "aarch64_asm_printer",
    srcs = glob([
        "lib/Target/AArch64/InstPrinter/*.c",
        "lib/Target/AArch64/InstPrinter/*.cpp",
        "lib/Target/AArch64/InstPrinter/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AArch64/InstPrinter/*.h",
        "include/llvm/Target/AArch64/InstPrinter/*.def",
        "include/llvm/Target/AArch64/InstPrinter/*.inc",
        "lib/Target/AArch64/InstPrinter/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AArch64"],
    deps = [
        ":aarch64_target_gen",
        ":aarch64_utils",
        ":config",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "aarch64_code_gen",
    srcs = glob([
        "lib/Target/AArch64/*.c",
        "lib/Target/AArch64/*.cpp",
        "lib/Target/AArch64/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AArch64/*.h",
        "include/llvm/Target/AArch64/*.def",
        "include/llvm/Target/AArch64/*.inc",
        "lib/Target/AArch64/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AArch64"],
    deps = [
        ":aarch64_asm_printer",
        ":aarch64_desc",
        ":aarch64_info",
        ":aarch64_utils",
        ":analysis",
        ":asm_printer",
        ":code_gen",
        ":config",
        ":core",
        ":global_i_sel",
        ":mc",
        ":scalar",
        ":selection_dag",
        ":support",
        ":target",
    ],
)

cc_library(
    name = "aarch64_desc",
    srcs = glob([
        "lib/Target/AArch64/MCTargetDesc/*.c",
        "lib/Target/AArch64/MCTargetDesc/*.cpp",
        "lib/Target/AArch64/MCTargetDesc/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AArch64/MCTargetDesc/*.h",
        "include/llvm/Target/AArch64/MCTargetDesc/*.def",
        "include/llvm/Target/AArch64/MCTargetDesc/*.inc",
        "lib/Target/AArch64/MCTargetDesc/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AArch64"],
    deps = [
        ":aarch64_asm_printer",
        ":aarch64_info",
        ":aarch64_target_gen",
        ":attributes_gen",
        ":config",
        ":intrinsics_gen",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "aarch64_disassembler",
    srcs = glob([
        "lib/Target/AArch64/Disassembler/*.c",
        "lib/Target/AArch64/Disassembler/*.cpp",
        "lib/Target/AArch64/Disassembler/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AArch64/Disassembler/*.h",
        "include/llvm/Target/AArch64/Disassembler/*.def",
        "include/llvm/Target/AArch64/Disassembler/*.inc",
        "lib/Target/AArch64/Disassembler/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AArch64"],
    deps = [
        ":aarch64_desc",
        ":aarch64_info",
        ":aarch64_utils",
        ":config",
        ":mc",
        ":mc_disassembler",
        ":support",
    ],
)

cc_library(
    name = "aarch64_info",
    srcs = glob([
        "lib/Target/AArch64/TargetInfo/*.c",
        "lib/Target/AArch64/TargetInfo/*.cpp",
        "lib/Target/AArch64/TargetInfo/*.inc",
        "lib/Target/AArch64/MCTargetDesc/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/AArch64/TargetInfo/*.h",
        "include/llvm/Target/AArch64/TargetInfo/*.def",
        "include/llvm/Target/AArch64/TargetInfo/*.inc",
        "lib/Target/AArch64/*.def",
        "lib/Target/AArch64/AArch64*.h",
        "lib/Target/AArch64/TargetInfo/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AArch64"],
    deps = [
        ":code_gen",
        ":config",
        ":support",
        ":target",
    ],
)

cc_library(
    name = "aarch64_utils",
    srcs = glob([
        "lib/Target/AArch64/Utils/*.c",
        "lib/Target/AArch64/Utils/*.cpp",
        "lib/Target/AArch64/Utils/*.inc",
        "lib/Target/AArch64/MCTargetDesc/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/AArch64/Utils/*.h",
        "include/llvm/Target/AArch64/Utils/*.def",
        "include/llvm/Target/AArch64/Utils/*.inc",
        "lib/Target/AArch64/Utils/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AArch64"],
    deps = [
        ":aarch64_target_gen",
        ":config",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "aggressive_inst_combine",
    srcs = glob([
        "lib/Transforms/AggressiveInstCombine/*.c",
        "lib/Transforms/AggressiveInstCombine/*.cpp",
        "lib/Transforms/AggressiveInstCombine/*.inc",
        "lib/Transforms/AggressiveInstCombine/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/AggressiveInstCombine/*.h",
        "include/llvm/Transforms/AggressiveInstCombine/*.def",
        "include/llvm/Transforms/AggressiveInstCombine/*.inc",
    ]),
    deps = [
        ":analysis",
        ":config",
        ":core",
        ":support",
        ":transform_utils",
    ],
)

cc_library(
    name = "analysis",
    srcs = glob([
        "lib/Analysis/*.c",
        "lib/Analysis/*.cpp",
        "lib/Analysis/*.inc",
        "include/llvm/Transforms/Utils/Local.h",
        "include/llvm/Transforms/Scalar.h",
        "lib/Analysis/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Analysis/*.h",
        "include/llvm/Analysis/*.def",
        "include/llvm/Analysis/*.inc",
    ]),
    deps = [
        ":binary_format",
        ":config",
        ":core",
        ":object",
        ":profile_data",
        ":support",
    ],
)

cc_library(
    name = "amdgpu_desc",
    srcs = glob([
        "lib/Target/AMDGPU/MCTargetDesc/*.c",
        "lib/Target/AMDGPU/MCTargetDesc/*.cpp",
        "lib/Target/AMDGPU/MCTargetDesc/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AMDGPU/MCTargetDesc/*.h",
        "include/llvm/Target/AMDGPU/MCTargetDesc/*.def",
        "include/llvm/Target/AMDGPU/MCTargetDesc/*.inc",
        "lib/Target/AMDGPU/MCTargetDesc/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AMDGPU"],
    deps = [
        ":amdgpu_asm_printer",
        ":amdgpu_info",
        ":amdgpu_utils",
        ":config",
        ":core",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "amdgpu_disassembler",
    srcs = glob([
        "lib/Target/AMDGPU/Disassembler/*.c",
        "lib/Target/AMDGPU/Disassembler/*.cpp",
        "lib/Target/AMDGPU/Disassembler/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AMDGPU/Disassembler/*.h",
        "include/llvm/Target/AMDGPU/Disassembler/*.def",
        "include/llvm/Target/AMDGPU/Disassembler/*.inc",
        "lib/Target/AMDGPU/Disassembler/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AMDGPU"],
    deps = [
        ":amdgpu_desc",
        ":amdgpu_info",
        ":amdgpu_utils",
        ":config",
        ":mc",
        ":mc_disassembler",
        ":support",
    ],
)

cc_library(
    name = "amdgpu_info",
    srcs = glob([
        "lib/Target/AMDGPU/TargetInfo/*.c",
        "lib/Target/AMDGPU/TargetInfo/*.cpp",
        "lib/Target/AMDGPU/TargetInfo/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AMDGPU/TargetInfo/*.h",
        "include/llvm/Target/AMDGPU/TargetInfo/*.def",
        "include/llvm/Target/AMDGPU/TargetInfo/*.inc",
        "lib/Target/AMDGPU/TargetInfo/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AMDGPU"],
    deps = [
        ":amdgpu_target_gen",
        ":config",
        ":core",
        ":support",
    ],
)

cc_library(
    name = "amdgpu_utils",
    srcs = glob([
        "lib/Target/AMDGPU/Utils/*.c",
        "lib/Target/AMDGPU/Utils/*.cpp",
        "lib/Target/AMDGPU/Utils/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AMDGPU/Utils/*.h",
        "include/llvm/Target/AMDGPU/Utils/*.def",
        "include/llvm/Target/AMDGPU/Utils/*.inc",
        "lib/Target/AMDGPU/Utils/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AMDGPU"],
    deps = [
        ":amdgpu_target_gen",
        ":config",
        ":core",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "amdgpu_asm_parser",
    srcs = glob([
        "lib/Target/AMDGPU/AsmParser/*.c",
        "lib/Target/AMDGPU/AsmParser/*.cpp",
        "lib/Target/AMDGPU/AsmParser/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AMDGPU/AsmParser/*.h",
        "include/llvm/Target/AMDGPU/AsmParser/*.def",
        "include/llvm/Target/AMDGPU/AsmParser/*.inc",
        "lib/Target/AMDGPU/AsmParser/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AMDGPU"],
    deps = [
        ":amdgpu_desc",
        ":amdgpu_info",
        ":amdgpu_utils",
        ":config",
        ":mc",
        ":mc_parser",
        ":support",
    ],
)

cc_library(
    name = "amdgpu_asm_printer",
    srcs = glob([
        "lib/Target/AMDGPU/InstPrinter/*.c",
        "lib/Target/AMDGPU/InstPrinter/*.cpp",
        "lib/Target/AMDGPU/InstPrinter/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AMDGPU/InstPrinter/*.h",
        "include/llvm/Target/AMDGPU/InstPrinter/*.def",
        "include/llvm/Target/AMDGPU/InstPrinter/*.inc",
        "lib/Target/AMDGPU/InstPrinter/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AMDGPU"],
    deps = [
        ":amdgpu_utils",
        ":config",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "amdgpu_code_gen",
    srcs = glob([
        "lib/Target/AMDGPU/*.c",
        "lib/Target/AMDGPU/*.cpp",
        "lib/Target/AMDGPU/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/AMDGPU/*.h",
        "include/llvm/Target/AMDGPU/*.def",
        "include/llvm/Target/AMDGPU/*.inc",
        "lib/Target/AMDGPU/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/AMDGPU"],
    deps = [
        ":amdgpu_asm_printer",
        ":amdgpu_desc",
        ":amdgpu_info",
        ":amdgpu_utils",
        ":analysis",
        ":asm_printer",
        ":code_gen",
        ":config",
        ":core",
        ":global_i_sel",
        ":ipo",
        ":mc",
        ":scalar",
        ":selection_dag",
        ":support",
        ":target",
        ":transform_utils",
        ":vectorize",
    ],
)

cc_library(
    name = "arm_asm_parser",
    srcs = glob([
        "lib/Target/ARM/AsmParser/*.c",
        "lib/Target/ARM/AsmParser/*.cpp",
        "lib/Target/ARM/AsmParser/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/ARM/AsmParser/*.h",
        "include/llvm/Target/ARM/AsmParser/*.def",
        "include/llvm/Target/ARM/AsmParser/*.inc",
        "lib/Target/ARM/AsmParser/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/ARM"],
    deps = [
        ":arm_desc",
        ":arm_info",
        ":arm_utils",
        ":config",
        ":mc",
        ":mc_parser",
        ":support",
    ],
)

cc_library(
    name = "arm_asm_printer",
    srcs = glob([
        "lib/Target/ARM/InstPrinter/*.c",
        "lib/Target/ARM/InstPrinter/*.cpp",
        "lib/Target/ARM/InstPrinter/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/ARM/InstPrinter/*.h",
        "include/llvm/Target/ARM/InstPrinter/*.def",
        "include/llvm/Target/ARM/InstPrinter/*.inc",
        "lib/Target/ARM/*.h",
        "lib/Target/ARM/InstPrinter/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/ARM"],
    deps = [
        ":arm_info",
        ":arm_target_gen",
        ":arm_utils",
        ":config",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "arm_code_gen",
    srcs = glob([
        "lib/Target/ARM/*.c",
        "lib/Target/ARM/*.cpp",
        "lib/Target/ARM/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/ARM/*.h",
        "include/llvm/Target/ARM/*.def",
        "include/llvm/Target/ARM/*.inc",
        "lib/Target/ARM/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/ARM"],
    deps = [
        ":analysis",
        ":arm_asm_printer",
        ":arm_desc",
        ":arm_info",
        ":arm_utils",
        ":asm_printer",
        ":code_gen",
        ":config",
        ":core",
        ":global_i_sel",
        ":mc",
        ":scalar",
        ":selection_dag",
        ":support",
        ":target",
    ],
)

cc_library(
    name = "arm_desc",
    srcs = glob([
        "lib/Target/ARM/MCTargetDesc/*.c",
        "lib/Target/ARM/MCTargetDesc/*.cpp",
        "lib/Target/ARM/MCTargetDesc/*.inc",
        "lib/Target/ARM/*.h",
        "include/llvm/CodeGen/GlobalISel/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/ARM/MCTargetDesc/*.h",
        "include/llvm/Target/ARM/MCTargetDesc/*.def",
        "include/llvm/Target/ARM/MCTargetDesc/*.inc",
        "lib/Target/ARM/MCTargetDesc/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/ARM"],
    deps = [
        ":arm_asm_printer",
        ":arm_info",
        ":arm_target_gen",
        ":attributes_gen",
        ":config",
        ":intrinsics_gen",
        ":mc",
        ":mc_disassembler",
        ":support",
    ],
)

cc_library(
    name = "arm_disassembler",
    srcs = glob([
        "lib/Target/ARM/Disassembler/*.c",
        "lib/Target/ARM/Disassembler/*.cpp",
        "lib/Target/ARM/Disassembler/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/ARM/Disassembler/*.h",
        "include/llvm/Target/ARM/Disassembler/*.def",
        "include/llvm/Target/ARM/Disassembler/*.inc",
        "lib/Target/ARM/Disassembler/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/ARM"],
    deps = [
        ":arm_desc",
        ":arm_info",
        ":arm_utils",
        ":config",
        ":mc_disassembler",
        ":support",
    ],
)

cc_library(
    name = "arm_info",
    srcs = glob([
        "lib/Target/ARM/TargetInfo/*.c",
        "lib/Target/ARM/TargetInfo/*.cpp",
        "lib/Target/ARM/TargetInfo/*.inc",
        "lib/Target/ARM/MCTargetDesc/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/ARM/TargetInfo/*.h",
        "include/llvm/Target/ARM/TargetInfo/*.def",
        "include/llvm/Target/ARM/TargetInfo/*.inc",
        "lib/Target/ARM/TargetInfo/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/ARM"],
    deps = [
        ":arm_target_gen",
        ":config",
        ":support",
        ":target",
    ],
)

cc_library(
    name = "arm_utils",
    srcs = glob([
        "lib/Target/ARM/Utils/*.c",
        "lib/Target/ARM/Utils/*.cpp",
        "lib/Target/ARM/Utils/*.inc",
        "lib/Target/ARM/MCTargetDesc/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/ARM/Utils/*.h",
        "include/llvm/Target/ARM/Utils/*.def",
        "include/llvm/Target/ARM/Utils/*.inc",
        "lib/Target/ARM/Utils/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/ARM"],
    deps = [
        ":arm_target_gen",
        ":config",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "asm_parser",
    srcs = glob([
        "lib/AsmParser/*.c",
        "lib/AsmParser/*.cpp",
        "lib/AsmParser/*.inc",
        "lib/AsmParser/*.h",
    ]),
    hdrs = glob([
        "include/llvm/AsmParser/*.h",
        "include/llvm/AsmParser/*.def",
        "include/llvm/AsmParser/*.inc",
    ]),
    deps = [
        ":binary_format",
        ":config",
        ":core",
        ":support",
    ],
)

cc_library(
    name = "asm_printer",
    srcs = glob([
        "lib/CodeGen/AsmPrinter/*.c",
        "lib/CodeGen/AsmPrinter/*.cpp",
        "lib/CodeGen/AsmPrinter/*.inc",
        "lib/CodeGen/AsmPrinter/*.h",
    ]),
    hdrs = glob([
        "include/llvm/CodeGen/AsmPrinter/*.h",
        "include/llvm/CodeGen/AsmPrinter/*.def",
        "include/llvm/CodeGen/AsmPrinter/*.inc",
        "lib/CodeGen/AsmPrinter/*.def",
    ]),
    deps = [
        ":analysis",
        ":binary_format",
        ":code_gen",
        ":config",
        ":core",
        ":debug_info_code_view",
        ":debug_info_msf",
        ":mc",
        ":mc_parser",
        ":support",
        ":target",
    ],
)

cc_library(
    name = "binary_format",
    srcs = glob([
        "lib/BinaryFormat/*.c",
        "lib/BinaryFormat/*.cpp",
        "lib/BinaryFormat/*.inc",
        "lib/BinaryFormat/*.h",
    ]),
    hdrs = glob([
        "include/llvm/BinaryFormat/*.h",
        "include/llvm/BinaryFormat/*.def",
        "include/llvm/BinaryFormat/*.inc",
        "include/llvm/BinaryFormat/ELFRelocs/*.def",
        "include/llvm/BinaryFormat/WasmRelocs/*.def",
    ]),
    deps = [
        ":config",
        ":support",
    ],
)

cc_library(
    name = "bit_reader",
    srcs = glob([
        "lib/Bitcode/Reader/*.c",
        "lib/Bitcode/Reader/*.cpp",
        "lib/Bitcode/Reader/*.inc",
        "lib/Bitcode/Reader/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Bitcode/Reader/*.h",
        "include/llvm/Bitcode/Reader/*.def",
        "include/llvm/Bitcode/Reader/*.inc",
        "include/llvm/Bitcode/BitstreamReader.h",
    ]),
    deps = [
        ":config",
        ":core",
        ":support",
    ],
)

cc_library(
    name = "bit_writer",
    srcs = glob([
        "lib/Bitcode/Writer/*.c",
        "lib/Bitcode/Writer/*.cpp",
        "lib/Bitcode/Writer/*.inc",
        "lib/Bitcode/Writer/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Bitcode/Writer/*.h",
        "include/llvm/Bitcode/Writer/*.def",
        "include/llvm/Bitcode/Writer/*.inc",
        "include/llvm/Bitcode/BitcodeWriter.h",
        "include/llvm/Bitcode/BitcodeWriterPass.h",
        "include/llvm/Bitcode/BitstreamWriter.h",
    ]),
    deps = [
        ":analysis",
        ":config",
        ":core",
        ":mc",
        ":object",
        ":support",
    ],
)

cc_library(
    name = "code_gen",
    srcs = glob([
        "lib/CodeGen/*.c",
        "lib/CodeGen/*.cpp",
        "lib/CodeGen/*.inc",
        "lib/CodeGen/*.h",
    ]),
    hdrs = glob([
        "include/llvm/CodeGen/*.h",
        "include/llvm/CodeGen/*.def",
        "include/llvm/CodeGen/*.inc",
        "include/llvm/CodeGen/**/*.h",
    ]),
    deps = [
        ":analysis",
        ":bit_reader",
        ":bit_writer",
        ":config",
        ":core",
        ":instrumentation",
        ":mc",
        ":profile_data",
        ":scalar",
        ":support",
        ":target",
        ":transform_utils",
    ],
)

cc_library(
    name = "core",
    srcs = glob([
        "lib/IR/*.c",
        "lib/IR/*.cpp",
        "lib/IR/*.inc",
        "include/llvm/Analysis/*.h",
        "include/llvm/Bitcode/BitcodeReader.h",
        "include/llvm/Bitcode/BitCodes.h",
        "include/llvm/Bitcode/LLVMBitCodes.h",
        "include/llvm/CodeGen/MachineValueType.h",
        "include/llvm/CodeGen/ValueTypes.h",
        "lib/IR/*.h",
    ]),
    hdrs = glob([
        "include/llvm/IR/*.h",
        "include/llvm/IR/*.def",
        "include/llvm/IR/*.inc",
        "include/llvm/*.h",
        "include/llvm/Analysis/*.def",
    ]),
    deps = [
        ":attributes_compat_gen",
        ":attributes_gen",
        ":binary_format",
        ":config",
        ":intrinsics_gen",
        ":support",
    ],
)

cc_library(
    name = "debug_info_code_view",
    srcs = glob([
        "lib/DebugInfo/CodeView/*.c",
        "lib/DebugInfo/CodeView/*.cpp",
        "lib/DebugInfo/CodeView/*.inc",
        "lib/DebugInfo/CodeView/*.h",
    ]),
    hdrs = glob([
        "include/llvm/DebugInfo/CodeView/*.h",
        "include/llvm/DebugInfo/CodeView/*.def",
        "include/llvm/DebugInfo/CodeView/*.inc",
    ]),
    deps = [
        ":binary_format",
        ":config",
        ":debug_info_msf",
        ":support",
    ],
)

cc_library(
    name = "debug_info_msf",
    srcs = glob([
        "lib/DebugInfo/MSF/*.c",
        "lib/DebugInfo/MSF/*.cpp",
        "lib/DebugInfo/MSF/*.inc",
        "lib/DebugInfo/MSF/*.h",
    ]),
    hdrs = glob([
        "include/llvm/DebugInfo/MSF/*.h",
        "include/llvm/DebugInfo/MSF/*.def",
        "include/llvm/DebugInfo/MSF/*.inc",
    ]),
    deps = [
        ":config",
        ":support",
    ],
)

cc_library(
    name = "demangle",
    srcs = glob([
        "lib/Demangle/*.c",
        "lib/Demangle/*.cpp",
        "lib/Demangle/*.inc",
        "lib/Demangle/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Demangle/*.h",
        "include/llvm/Demangle/*.def",
        "include/llvm/Demangle/*.inc",
    ]),
    deps = [":config"],
)

cc_library(
    name = "execution_engine",
    srcs = glob([
        "lib/ExecutionEngine/*.c",
        "lib/ExecutionEngine/*.cpp",
        "lib/ExecutionEngine/*.inc",
        "lib/ExecutionEngine/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ExecutionEngine/*.h",
        "include/llvm/ExecutionEngine/*.def",
        "include/llvm/ExecutionEngine/*.inc",
    ]),
    deps = [
        ":config",
        ":core",
        ":mc",
        ":object",
        ":runtime_dyld",
        ":support",
        ":target",
    ],
)

cc_library(
    name = "global_i_sel",
    srcs = glob([
        "lib/CodeGen/GlobalISel/*.c",
        "lib/CodeGen/GlobalISel/*.cpp",
        "lib/CodeGen/GlobalISel/*.inc",
        "lib/CodeGen/GlobalISel/*.h",
    ]),
    hdrs = glob([
        "include/llvm/CodeGen/GlobalISel/*.h",
        "include/llvm/CodeGen/GlobalISel/*.def",
        "include/llvm/CodeGen/GlobalISel/*.inc",
    ]),
    deps = [
        ":analysis",
        ":code_gen",
        ":config",
        ":core",
        ":mc",
        ":support",
        ":target",
        ":transform_utils",
    ],
)

cc_library(
    name = "instrumentation",
    srcs = glob([
        "lib/Transforms/Instrumentation/*.c",
        "lib/Transforms/Instrumentation/*.cpp",
        "lib/Transforms/Instrumentation/*.inc",
        "lib/Transforms/Instrumentation/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/Instrumentation/*.h",
        "include/llvm/Transforms/Instrumentation/*.def",
        "include/llvm/Transforms/Instrumentation/*.inc",
        "include/llvm/Transforms/GCOVProfiler.h",
        "include/llvm/Transforms/Instrumentation.h",
        "include/llvm/Transforms/InstrProfiling.h",
        "include/llvm/Transforms/PGOInstrumentation.h",
    ]),
    deps = [
        ":analysis",
        ":config",
        ":core",
        ":mc",
        ":profile_data",
        ":support",
        ":transform_utils",
    ],
)

cc_library(
    name = "inst_combine",
    srcs = glob([
        "lib/Transforms/InstCombine/*.c",
        "lib/Transforms/InstCombine/*.cpp",
        "lib/Transforms/InstCombine/*.inc",
        "lib/Transforms/InstCombine/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/InstCombine/*.h",
        "include/llvm/Transforms/InstCombine/*.def",
        "include/llvm/Transforms/InstCombine/*.inc",
    ]),
    deps = [
        ":analysis",
        ":config",
        ":core",
        ":support",
        ":transform_utils",
    ],
)

cc_library(
    name = "ipo",
    srcs = glob([
        "lib/Transforms/IPO/*.c",
        "lib/Transforms/IPO/*.cpp",
        "lib/Transforms/IPO/*.inc",
        "include/llvm/Transforms/SampleProfile.h",
        "include/llvm-c/Transforms/IPO.h",
        "include/llvm-c/Transforms/PassManagerBuilder.h",
        "lib/Transforms/IPO/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/IPO/*.h",
        "include/llvm/Transforms/IPO/*.def",
        "include/llvm/Transforms/IPO/*.inc",
    ]),
    deps = [
        ":aggressive_inst_combine",
        ":analysis",
        ":bit_reader",
        ":bit_writer",
        ":config",
        ":core",
        ":inst_combine",
        ":instrumentation",
        ":ir_reader",
        ":linker",
        ":object",
        ":profile_data",
        ":scalar",
        ":support",
        ":transform_utils",
        ":vectorize",
    ],
)

cc_library(
    name = "ir_reader",
    srcs = glob([
        "lib/IRReader/*.c",
        "lib/IRReader/*.cpp",
        "lib/IRReader/*.inc",
        "lib/IRReader/*.h",
    ]),
    hdrs = glob([
        "include/llvm/IRReader/*.h",
        "include/llvm/IRReader/*.def",
        "include/llvm/IRReader/*.inc",
    ]),
    deps = [
        ":asm_parser",
        ":bit_reader",
        ":config",
        ":core",
        ":support",
    ],
)

cc_library(
    name = "linker",
    srcs = glob([
        "lib/Linker/*.c",
        "lib/Linker/*.cpp",
        "lib/Linker/*.inc",
        "lib/Linker/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Linker/*.h",
        "include/llvm/Linker/*.def",
        "include/llvm/Linker/*.inc",
    ]),
    deps = [
        ":config",
        ":core",
        ":support",
        ":transform_utils",
    ],
)

cc_library(
    name = "mc",
    srcs = glob([
        "lib/MC/*.c",
        "lib/MC/*.cpp",
        "lib/MC/*.inc",
        "lib/MC/*.h",
    ]),
    hdrs = glob([
        "include/llvm/MC/*.h",
        "include/llvm/MC/*.def",
        "include/llvm/MC/*.inc",
    ]),
    deps = [
        ":binary_format",
        ":config",
        ":debug_info_code_view",
        ":support",
    ],
)

cc_library(
    name = "mc_disassembler",
    srcs = glob([
        "lib/MC/MCDisassembler/*.c",
        "lib/MC/MCDisassembler/*.cpp",
        "lib/MC/MCDisassembler/*.inc",
        "lib/MC/MCDisassembler/*.h",
    ]),
    hdrs = glob([
        "include/llvm/MC/MCDisassembler/*.h",
        "include/llvm/MC/MCDisassembler/*.def",
        "include/llvm/MC/MCDisassembler/*.inc",
    ]),
    deps = [
        ":config",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "mc_parser",
    srcs = glob([
        "lib/MC/MCParser/*.c",
        "lib/MC/MCParser/*.cpp",
        "lib/MC/MCParser/*.inc",
        "lib/MC/MCParser/*.h",
    ]),
    hdrs = glob([
        "include/llvm/MC/MCParser/*.h",
        "include/llvm/MC/MCParser/*.def",
        "include/llvm/MC/MCParser/*.inc",
    ]),
    deps = [
        ":config",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "nvptx_asm_printer",
    srcs = glob([
        "lib/Target/NVPTX/InstPrinter/*.c",
        "lib/Target/NVPTX/InstPrinter/*.cpp",
        "lib/Target/NVPTX/InstPrinter/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/NVPTX/InstPrinter/*.h",
        "include/llvm/Target/NVPTX/InstPrinter/*.def",
        "include/llvm/Target/NVPTX/InstPrinter/*.inc",
        "lib/Target/NVPTX/InstPrinter/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/NVPTX"],
    deps = [
        "nvptx_target_gen",
        ":attributes_gen",
        ":config",
        ":mc",
        ":nvptx_info",
        ":support",
    ],
)

cc_library(
    name = "nvptx_code_gen",
    srcs = glob([
        "lib/Target/NVPTX/*.c",
        "lib/Target/NVPTX/*.cpp",
        "lib/Target/NVPTX/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/NVPTX/*.h",
        "include/llvm/Target/NVPTX/*.def",
        "include/llvm/Target/NVPTX/*.inc",
        "lib/Target/NVPTX/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/NVPTX"],
    deps = [
        ":analysis",
        ":asm_printer",
        ":code_gen",
        ":config",
        ":core",
        ":ipo",
        ":mc",
        ":nvptx_asm_printer",
        ":nvptx_desc",
        ":nvptx_info",
        ":scalar",
        ":selection_dag",
        ":support",
        ":target",
        ":transform_utils",
        ":vectorize",
    ],
)

cc_library(
    name = "nvptx_desc",
    srcs = glob([
        "lib/Target/NVPTX/MCTargetDesc/*.c",
        "lib/Target/NVPTX/MCTargetDesc/*.cpp",
        "lib/Target/NVPTX/MCTargetDesc/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/NVPTX/MCTargetDesc/*.h",
        "include/llvm/Target/NVPTX/MCTargetDesc/*.def",
        "include/llvm/Target/NVPTX/MCTargetDesc/*.inc",
        "lib/Target/NVPTX/MCTargetDesc/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/NVPTX"],
    deps = [
        "nvptx_target_gen",
        ":config",
        ":mc",
        ":nvptx_asm_printer",
        ":nvptx_info",
        ":support",
    ],
)

cc_library(
    name = "nvptx_info",
    srcs = glob([
        "lib/Target/NVPTX/TargetInfo/*.c",
        "lib/Target/NVPTX/TargetInfo/*.cpp",
        "lib/Target/NVPTX/TargetInfo/*.inc",
        "lib/Target/NVPTX/MCTargetDesc/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/NVPTX/TargetInfo/*.h",
        "include/llvm/Target/NVPTX/TargetInfo/*.def",
        "include/llvm/Target/NVPTX/TargetInfo/*.inc",
        "lib/Target/NVPTX/NVPTX.h",
        "lib/Target/NVPTX/TargetInfo/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/NVPTX"],
    deps = [
        "nvptx_target_gen",
        ":attributes_gen",
        ":config",
        ":core",
        ":support",
        ":target",
    ],
)

cc_library(
    name = "object",
    srcs = glob([
        "lib/Object/*.c",
        "lib/Object/*.cpp",
        "lib/Object/*.inc",
        "lib/Object/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Object/*.h",
        "include/llvm/Object/*.def",
        "include/llvm/Object/*.inc",
    ]),
    deps = [
        ":binary_format",
        ":bit_reader",
        ":config",
        ":core",
        ":mc",
        ":mc_parser",
        ":support",
    ],
)

cc_library(
    name = "objc_arc",
    srcs = glob([
        "lib/Transforms/ObjCARC/*.c",
        "lib/Transforms/ObjCARC/*.cpp",
        "lib/Transforms/ObjCARC/*.inc",
        "include/llvm/Transforms/ObjCARC.h",
        "lib/Transforms/ObjCARC/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/ObjCARC/*.h",
        "include/llvm/Transforms/ObjCARC/*.def",
        "include/llvm/Transforms/ObjCARC/*.inc",
    ]),
    deps = [
        ":analysis",
        ":config",
        ":core",
        ":support",
        ":transform_utils",
    ],
)

cc_library(
    name = "orc_jit",
    srcs = glob([
        "lib/ExecutionEngine/Orc/*.c",
        "lib/ExecutionEngine/Orc/*.cpp",
        "lib/ExecutionEngine/Orc/*.inc",
        "lib/ExecutionEngine/Orc/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ExecutionEngine/Orc/*.h",
        "include/llvm/ExecutionEngine/Orc/*.def",
        "include/llvm/ExecutionEngine/Orc/*.inc",
    ]),
    deps = [
        ":config",
        ":core",
        ":execution_engine",
        ":object",
        ":runtime_dyld",
        ":support",
        ":transform_utils",
    ],
)

cc_library(
    name = "powerpc_asm_parser",
    srcs = glob([
        "lib/Target/PowerPC/AsmParser/*.c",
        "lib/Target/PowerPC/AsmParser/*.cpp",
        "lib/Target/PowerPC/AsmParser/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/PowerPC/AsmParser/*.h",
        "include/llvm/Target/PowerPC/AsmParser/*.def",
        "include/llvm/Target/PowerPC/AsmParser/*.inc",
        "lib/Target/PowerPC/AsmParser/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/PowerPC"],
    deps = [
        ":config",
        ":mc",
        ":mc_parser",
        ":powerpc_desc",
        ":powerpc_info",
        ":support",
    ],
)

cc_library(
    name = "powerpc_asm_printer",
    srcs = glob([
        "lib/Target/PowerPC/InstPrinter/*.c",
        "lib/Target/PowerPC/InstPrinter/*.cpp",
        "lib/Target/PowerPC/InstPrinter/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/PowerPC/InstPrinter/*.h",
        "include/llvm/Target/PowerPC/InstPrinter/*.def",
        "include/llvm/Target/PowerPC/InstPrinter/*.inc",
        "lib/Target/PowerPC/InstPrinter/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/PowerPC"],
    deps = [
        ":attributes_gen",
        ":config",
        ":intrinsics_gen",
        ":mc",
        ":powerpc_info",
        ":powerpc_target_gen",
        ":support",
    ],
)

cc_library(
    name = "powerpc_code_gen",
    srcs = glob([
        "lib/Target/PowerPC/*.c",
        "lib/Target/PowerPC/*.cpp",
        "lib/Target/PowerPC/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/PowerPC/*.h",
        "include/llvm/Target/PowerPC/*.def",
        "include/llvm/Target/PowerPC/*.inc",
        "lib/Target/PowerPC/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/PowerPC"],
    deps = [
        ":analysis",
        ":asm_printer",
        ":code_gen",
        ":config",
        ":core",
        ":mc",
        ":powerpc_asm_printer",
        ":powerpc_desc",
        ":powerpc_info",
        ":scalar",
        ":selection_dag",
        ":support",
        ":target",
        ":transform_utils",
    ],
)

cc_library(
    name = "powerpc_desc",
    srcs = glob([
        "lib/Target/PowerPC/MCTargetDesc/*.c",
        "lib/Target/PowerPC/MCTargetDesc/*.cpp",
        "lib/Target/PowerPC/MCTargetDesc/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/PowerPC/MCTargetDesc/*.h",
        "include/llvm/Target/PowerPC/MCTargetDesc/*.def",
        "include/llvm/Target/PowerPC/MCTargetDesc/*.inc",
        "lib/Target/PowerPC/MCTargetDesc/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/PowerPC"],
    deps = [
        ":attributes_gen",
        ":config",
        ":intrinsics_gen",
        ":mc",
        ":powerpc_asm_printer",
        ":powerpc_info",
        ":powerpc_target_gen",
        ":support",
    ],
)

cc_library(
    name = "powerpc_disassembler",
    srcs = glob([
        "lib/Target/PowerPC/Disassembler/*.c",
        "lib/Target/PowerPC/Disassembler/*.cpp",
        "lib/Target/PowerPC/Disassembler/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/PowerPC/Disassembler/*.h",
        "include/llvm/Target/PowerPC/Disassembler/*.def",
        "include/llvm/Target/PowerPC/Disassembler/*.inc",
        "lib/Target/PowerPC/Disassembler/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/PowerPC"],
    deps = [
        ":config",
        ":mc_disassembler",
        ":powerpc_info",
        ":support",
    ],
)

cc_library(
    name = "powerpc_info",
    srcs = glob([
        "lib/Target/PowerPC/TargetInfo/*.c",
        "lib/Target/PowerPC/TargetInfo/*.cpp",
        "lib/Target/PowerPC/TargetInfo/*.inc",
        "lib/Target/PowerPC/MCTargetDesc/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/PowerPC/TargetInfo/*.h",
        "include/llvm/Target/PowerPC/TargetInfo/*.def",
        "include/llvm/Target/PowerPC/TargetInfo/*.inc",
        "lib/Target/PowerPC/PPC*.h",
        "lib/Target/PowerPC/TargetInfo/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/PowerPC"],
    deps = [
        ":attributes_gen",
        ":config",
        ":core",
        ":intrinsics_gen",
        ":powerpc_target_gen",
        ":support",
        ":target",
    ],
)

cc_library(
    name = "profile_data",
    srcs = glob([
        "lib/ProfileData/*.c",
        "lib/ProfileData/*.cpp",
        "lib/ProfileData/*.inc",
        "lib/ProfileData/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ProfileData/*.h",
        "include/llvm/ProfileData/*.def",
        "include/llvm/ProfileData/*.inc",
    ]),
    deps = [
        ":config",
        ":core",
        ":support",
    ],
)

cc_library(
    name = "runtime_dyld",
    srcs = glob([
        "lib/ExecutionEngine/RuntimeDyld/*.c",
        "lib/ExecutionEngine/RuntimeDyld/*.cpp",
        "lib/ExecutionEngine/RuntimeDyld/*.inc",
        "include/llvm/ExecutionEngine/JITSymbol.h",
        "include/llvm/ExecutionEngine/RTDyldMemoryManager.h",
        "lib/ExecutionEngine/RuntimeDyld/*.h",
        "lib/ExecutionEngine/RuntimeDyld/Targets/*.h",
        "lib/ExecutionEngine/RuntimeDyld/Targets/*.cpp",
        "lib/ExecutionEngine/RuntimeDyld/*.h",
    ]),
    hdrs = glob([
        "include/llvm/ExecutionEngine/RuntimeDyld/*.h",
        "include/llvm/ExecutionEngine/RuntimeDyld/*.def",
        "include/llvm/ExecutionEngine/RuntimeDyld/*.inc",
        "include/llvm/DebugInfo/DIContext.h",
        "include/llvm/ExecutionEngine/RTDyldMemoryManager.h",
        "include/llvm/ExecutionEngine/RuntimeDyld*.h",
    ]),
    deps = [
        ":config",
        ":mc",
        ":mc_disassembler",
        ":object",
        ":support",
    ],
)

cc_library(
    name = "scalar",
    srcs = glob([
        "lib/Transforms/Scalar/*.c",
        "lib/Transforms/Scalar/*.cpp",
        "lib/Transforms/Scalar/*.inc",
        "include/llvm-c/Transforms/Scalar.h",
        "include/llvm/Transforms/Scalar.h",
        "include/llvm/Target/TargetMachine.h",
        "lib/Transforms/Scalar/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/Scalar/*.h",
        "include/llvm/Transforms/Scalar/*.def",
        "include/llvm/Transforms/Scalar/*.inc",
        "include/llvm/Transforms/IPO.h",
        "include/llvm/Transforms/IPO/SCCP.h",
    ]),
    deps = [
        ":aggressive_inst_combine",
        ":analysis",
        ":config",
        ":core",
        ":inst_combine",
        ":support",
        ":target",
        ":transform_utils",
    ],
)

cc_library(
    name = "selection_dag",
    srcs = glob([
        "lib/CodeGen/SelectionDAG/*.c",
        "lib/CodeGen/SelectionDAG/*.cpp",
        "lib/CodeGen/SelectionDAG/*.inc",
        "lib/CodeGen/SelectionDAG/*.h",
    ]),
    hdrs = glob([
        "include/llvm/CodeGen/SelectionDAG/*.h",
        "include/llvm/CodeGen/SelectionDAG/*.def",
        "include/llvm/CodeGen/SelectionDAG/*.inc",
    ]),
    deps = [
        ":analysis",
        ":code_gen",
        ":config",
        ":core",
        ":mc",
        ":support",
        ":target",
        ":transform_utils",
    ],
)

cc_library(
    name = "support",
    srcs = glob([
        "lib/Support/*.c",
        "lib/Support/*.cpp",
        "lib/Support/*.inc",
        "lib/Support/Unix/*.inc",
        "lib/Support/Unix/*.h",
        "include/llvm-c/*.h",
        "include/llvm/CodeGen/MachineValueType.h",
        "include/llvm/BinaryFormat/COFF.h",
        "include/llvm/BinaryFormat/MachO.h",
        "lib/Support/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Support/*.h",
        "include/llvm/Support/*.def",
        "include/llvm/Support/*.inc",
        "include/llvm/ADT/*.h",
        "include/llvm/Support/ELFRelocs/*.def",
        "include/llvm/Support/WasmRelocs/*.def",
    ]) + [
        "include/llvm/BinaryFormat/MachO.def",
        "include/llvm/Support/VCSRevision.h",
        "include/llvm/ExecutionEngine/ObjectMemoryBuffer.h",
    ],
    deps = [
        ":config",
        ":demangle",
        "@zlib_archive//:zlib",
    ],
)

cc_library(
    name = "table_gen",
    srcs = glob([
        "lib/TableGen/*.c",
        "lib/TableGen/*.cpp",
        "lib/TableGen/*.inc",
        "include/llvm/CodeGen/*.h",
        "lib/TableGen/*.h",
    ]),
    hdrs = glob([
        "include/llvm/TableGen/*.h",
        "include/llvm/TableGen/*.def",
        "include/llvm/TableGen/*.inc",
        "include/llvm/Target/*.def",
    ]),
    deps = [
        ":config",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "target",
    srcs = glob([
        "lib/Target/*.c",
        "lib/Target/*.cpp",
        "lib/Target/*.inc",
        "include/llvm/CodeGen/*.h",
        "include/llvm-c/Initialization.h",
        "include/llvm-c/Target.h",
        "lib/Target/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/*.h",
        "include/llvm/Target/*.def",
        "include/llvm/Target/*.inc",
        "include/llvm/CodeGen/*.def",
        "include/llvm/CodeGen/*.inc",
    ]),
    deps = [
        ":analysis",
        ":config",
        ":core",
        ":mc",
        ":support",
    ],
)

cc_library(
    name = "transform_utils",
    srcs = glob([
        "lib/Transforms/Utils/*.c",
        "lib/Transforms/Utils/*.cpp",
        "lib/Transforms/Utils/*.inc",
        "include/llvm/Transforms/IPO.h",
        "include/llvm/Transforms/Scalar.h",
        "lib/Transforms/Utils/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/Utils/*.h",
        "include/llvm/Transforms/Utils/*.def",
        "include/llvm/Transforms/Utils/*.inc",
    ]),
    deps = [
        ":analysis",
        ":config",
        ":core",
        ":support",
    ],
)

cc_library(
    name = "vectorize",
    srcs = glob([
        "lib/Transforms/Vectorize/*.c",
        "lib/Transforms/Vectorize/*.cpp",
        "lib/Transforms/Vectorize/*.inc",
        "include/llvm-c/Transforms/Vectorize.h",
        "lib/Transforms/Vectorize/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Transforms/Vectorize/*.h",
        "include/llvm/Transforms/Vectorize/*.def",
        "include/llvm/Transforms/Vectorize/*.inc",
        "include/llvm/Transforms/Vectorize.h",
    ]),
    deps = [
        ":analysis",
        ":config",
        ":core",
        ":scalar",
        ":support",
        ":transform_utils",
    ],
)

cc_library(
    name = "x86_asm_parser",
    srcs = glob([
        "lib/Target/X86/AsmParser/*.c",
        "lib/Target/X86/AsmParser/*.cpp",
        "lib/Target/X86/AsmParser/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/X86/AsmParser/*.h",
        "include/llvm/Target/X86/AsmParser/*.def",
        "include/llvm/Target/X86/AsmParser/*.inc",
        "lib/Target/X86/AsmParser/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/X86"],
    deps = [
        ":config",
        ":mc",
        ":mc_parser",
        ":support",
        ":x86_asm_printer",
        ":x86_desc",
        ":x86_info",
    ],
)

cc_library(
    name = "x86_asm_printer",
    srcs = glob([
        "lib/Target/X86/InstPrinter/*.c",
        "lib/Target/X86/InstPrinter/*.cpp",
        "lib/Target/X86/InstPrinter/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/X86/InstPrinter/*.h",
        "include/llvm/Target/X86/InstPrinter/*.def",
        "include/llvm/Target/X86/InstPrinter/*.inc",
        "lib/Target/X86/InstPrinter/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/X86"],
    deps = [
        ":config",
        ":mc",
        ":support",
        ":x86_info",
        ":x86_target_gen",
        ":x86_utils",
    ],
)

cc_library(
    name = "x86_code_gen",
    srcs = glob([
        "lib/Target/X86/*.c",
        "lib/Target/X86/*.cpp",
        "lib/Target/X86/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/X86/*.h",
        "include/llvm/Target/X86/*.def",
        "include/llvm/Target/X86/*.inc",
        "lib/Target/X86/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/X86"],
    deps = [
        ":analysis",
        ":asm_printer",
        ":code_gen",
        ":config",
        ":core",
        ":global_i_sel",
        ":mc",
        ":selection_dag",
        ":support",
        ":target",
        ":x86_asm_printer",
        ":x86_defs",
        ":x86_desc",
        ":x86_info",
        ":x86_utils",
    ],
)

cc_library(
    name = "x86_desc",
    srcs = glob([
        "lib/Target/X86/MCTargetDesc/*.c",
        "lib/Target/X86/MCTargetDesc/*.cpp",
        "lib/Target/X86/MCTargetDesc/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/X86/MCTargetDesc/*.h",
        "include/llvm/Target/X86/MCTargetDesc/*.def",
        "include/llvm/Target/X86/MCTargetDesc/*.inc",
        "lib/Target/X86/MCTargetDesc/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/X86"],
    deps = [
        ":config",
        ":mc",
        ":mc_disassembler",
        ":object",
        ":support",
        ":x86_asm_printer",
        ":x86_info",
    ],
)

cc_library(
    name = "x86_disassembler",
    srcs = glob([
        "lib/Target/X86/Disassembler/*.c",
        "lib/Target/X86/Disassembler/*.cpp",
        "lib/Target/X86/Disassembler/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/X86/Disassembler/*.h",
        "include/llvm/Target/X86/Disassembler/*.def",
        "include/llvm/Target/X86/Disassembler/*.inc",
        "lib/Target/X86/Disassembler/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/X86"],
    deps = [
        ":config",
        ":mc_disassembler",
        ":support",
        ":x86_info",
    ],
)

cc_library(
    name = "x86_info",
    srcs = glob([
        "lib/Target/X86/TargetInfo/*.c",
        "lib/Target/X86/TargetInfo/*.cpp",
        "lib/Target/X86/TargetInfo/*.inc",
        "lib/Target/X86/MCTargetDesc/*.h",
    ]),
    hdrs = glob([
        "include/llvm/Target/X86/TargetInfo/*.h",
        "include/llvm/Target/X86/TargetInfo/*.def",
        "include/llvm/Target/X86/TargetInfo/*.inc",
        "lib/Target/X86/TargetInfo/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/X86"],
    deps = [
        ":config",
        ":mc",
        ":support",
        ":x86_target_gen",
    ],
)

cc_library(
    name = "x86_utils",
    srcs = glob([
        "lib/Target/X86/Utils/*.c",
        "lib/Target/X86/Utils/*.cpp",
        "lib/Target/X86/Utils/*.inc",
    ]),
    hdrs = glob([
        "include/llvm/Target/X86/Utils/*.h",
        "include/llvm/Target/X86/Utils/*.def",
        "include/llvm/Target/X86/Utils/*.inc",
        "lib/Target/X86/Utils/*.h",
    ]),
    copts = ["-Iexternal/llvm/lib/Target/X86"],
    deps = [
        ":code_gen",
        ":config",
        ":core",
        ":support",
    ],
)
