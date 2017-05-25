major_version: "local"
minor_version: ""
default_target_cpu: "same_as_host"

default_toolchain {
  cpu: "k8"
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
  tool_path { name: "gcc" path: "computecpp" }
  # Use "-std=c++11" for nvcc. For consistency, force both the host compiler
  # and the device compiler to use "-std=c++11".
  cxx_flag: "-std=c++11"
  linker_flag: "-Wl,-no-as-needed"
  linker_flag: "-lstdc++"
  linker_flag: "-B/usr/bin/"

  # TODO(bazel-team): In theory, the path here ought to exactly match the path
  # used by gcc. That works because bazel currently doesn't track files at
  # absolute locations and has no remote execution, yet. However, this will need
  # to be fixed, maybe with auto-detection?
  cxx_builtin_include_directory: "/usr/lib/gcc/"
  cxx_builtin_include_directory: "/usr/lib"
  cxx_builtin_include_directory: "/usr/lib64"
  cxx_builtin_include_directory: "/usr/local/include"
  cxx_builtin_include_directory: "/usr/include"

  cxx_builtin_include_directory: "%{computecpp_toolkit_path}"

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

  # Make C++ compilation deterministic. Use linkstamping instead of these
  # compiler symbols.
  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""

  # All warnings are enabled. Maybe enable -Werror as well?
  compiler_flag: "-Wall"

  # Enable SSE instructions by default
  compiler_flag: "-msse3"

  # Anticipated future default.
  linker_flag: "-Wl,-no-as-needed"
  # Stamp the binary with a unique identifier.
  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"

  linking_mode_flags { mode: DYNAMIC }

  compilation_mode_flags {
    mode: FASTBUILD
    compiler_flag: "-O0"
  }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "-g"
  }

  compilation_mode_flags {
    mode: OPT
    compiler_flag: "-g0"
    compiler_flag: "-O2"
    compiler_flag: "-DNDEBUG"
  }
}
