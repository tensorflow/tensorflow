major_version: "local"
minor_version: ""
default_target_cpu: "same_as_host"

default_toolchain {
  cpu: "k8"
  toolchain_identifier: "local_linux"
}
default_toolchain {
  cpu: "piii"
  toolchain_identifier: "local_linux"
}
default_toolchain {
  cpu: "arm"
  toolchain_identifier: "local_linux"
}
default_toolchain {
  cpu: "ppc"
  toolchain_identifier: "local_linux"
}

toolchain {
  abi_version: "local"
  abi_libc_version: "local"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "local"
  needsPic: true
  supports_gold_linker: false
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false
  supports_thin_archives: false
  target_libc: "local"
  target_cpu: "local"
  target_system_name: "local"
  toolchain_identifier: "local_linux"

  tool_path { name: "ar" path: "/usr/bin/ar" }
  tool_path { name: "compat-ld" path: "/usr/bin/ld" }
  tool_path { name: "cpp" path: "/usr/bin/cpp" }
  tool_path { name: "dwp" path: "/usr/bin/dwp" }
  # As part of the TensorFlow release, we place some ROCm-related compilation
  # files in @local_config_rocm//crosstool/clang/bin, and this relative
  # path, combined with the rest of our Bazel configuration causes our
  # compilation to use those files.
  tool_path { name: "gcc" path: "clang/bin/crosstool_wrapper_driver_rocm" }
  # Use "-std=c++11" for hipcc. For consistency, force both the host compiler
  # and the device compiler to use "-std=c++11".
  cxx_flag: "-std=c++11"
  linker_flag: "-Wl,-no-as-needed"
  linker_flag: "-lstdc++"
  #linker_flag: "-B/usr/bin/"
  linker_flag: "-B/opt/rocm/hcc/compiler/bin"

%{host_compiler_includes}
  tool_path { name: "gcov" path: "/usr/bin/gcov" }

  # C(++) compiles invoke the compiler (as that is the one knowing where
  # to find libraries), but we provide LD so other rules can invoke the linker.
  tool_path { name: "ld" path: "/usr/bin/ld" }

  tool_path { name: "nm" path: "/usr/bin/nm" }
  tool_path { name: "objcopy" path: "/usr/bin/objcopy" }
  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"
  tool_path { name: "objdump" path: "/usr/bin/objdump" }
  tool_path { name: "strip" path: "/usr/bin/strip" }

  # Anticipated future default.
  unfiltered_cxx_flag: "-no-canonical-prefixes"

  # Make C++ compilation deterministic. Use linkstamping instead of these
  # compiler symbols.
  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""
  unfiltered_cxx_flag: "-D__HIP_PLATFORM_HCC__"
  # The macro EIGEN_USE_HIP is used to tell Eigen to use the HIP platform headers
  # It needs to be always set when compiling Eigen headers
  # (irrespective of whether the source file is being compiled via HIPCC)
  # so adding -DEIGEN_USE_HIP as a default CXX flag here
  unfiltered_cxx_flag: "-DEIGEN_USE_HIP"

    
  # Security hardening on by default.
  # Conservative choice; -D_FORTIFY_SOURCE=2 may be unsafe in some cases.
  # We need to undef it before redefining it as some distributions now have
  # it enabled by default.
  #compiler_flag: "-U_FORTIFY_SOURCE"
  #compiler_flag: "-D_FORTIFY_SOURCE=1"
  #compiler_flag: "-fstack-protector"
  #compiler_flag: "-fPIE"
  #linker_flag: "-pie"
  #linker_flag: "-Wl,-z,relro,-z,now"

  # Enable coloring even if there's no attached terminal. Bazel removes the
  # escape sequences if --nocolor is specified. This isn't supported by gcc
  # on Ubuntu 14.04.
  # compiler_flag: "-fcolor-diagnostics"

  # All warnings are enabled. Maybe enable -Werror as well?
  compiler_flag: "-Wall"
  # Enable a few more warnings that aren't part of -Wall.
  compiler_flag: "-Wunused-but-set-parameter"
  # But disable some that are problematic.
  compiler_flag: "-Wno-free-nonheap-object" # has false positives

  # Keep stack frames for debugging, even in opt mode.
  compiler_flag: "-fno-omit-frame-pointer"

  # Anticipated future default.
  linker_flag: "-no-canonical-prefixes"
  unfiltered_cxx_flag: "-fno-canonical-system-headers"
  # Have gcc return the exit code from ld.
  linker_flag: "-pass-exit-codes"
  # Stamp the binary with a unique identifier.
  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"
  # Gold linker only? Can we enable this by default?
  # linker_flag: "-Wl,--warn-execstack"
  # linker_flag: "-Wl,--detect-odr-violations"

  # Include directory for ROCm headers.
%{rocm_include_path}

  compilation_mode_flags {
    mode: DBG
    # Enable debug symbols.
    compiler_flag: "-g"
  }
  compilation_mode_flags {
    mode: OPT

    # No debug symbols.
    # Maybe we should enable https://gcc.gnu.org/wiki/DebugFission for opt or
    # even generally? However, that can't happen here, as it requires special
    # handling in Bazel.
    compiler_flag: "-g0"

    # Conservative choice for -O
    # -O3 can increase binary size and even slow down the resulting binaries.
    # Profile first and / or use FDO if you need better performance than this.
    compiler_flag: "-O2"

    # Disable assertions
    compiler_flag: "-DNDEBUG"

    # Removal of unused code and data at link time (can this increase binary size in some cases?).
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"

    # With hipcc -fno-gpu-rdc, objects and libraries would contain a .kernel
    # section which would be removed by ld. Therefore --gc-sections shall be
    # abolished here.
    #linker_flag: "-Wl,--gc-sections"
  }
  linking_mode_flags { mode: DYNAMIC }
}
