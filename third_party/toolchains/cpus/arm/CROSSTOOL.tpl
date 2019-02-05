major_version: "local"
minor_version: ""
default_target_cpu: "same_as_host"

default_toolchain {
  cpu: "darwin"
  toolchain_identifier: "local_darwin"
}
default_toolchain {
  cpu: "freebsd"
  toolchain_identifier: "local_freebsd"
}
default_toolchain {
  cpu: "x64_windows"
  toolchain_identifier: "local_windows_msys64"
}
default_toolchain {
  cpu: "x64_windows_msvc"
  toolchain_identifier: "vc_14_0_x64"
}

toolchain {
  abi_version: "armeabi"
  abi_libc_version: "armeabi"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "armeabi"
  needsPic: true
  supports_gold_linker: false
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false
  target_libc: "armeabi"
  target_cpu: "armeabi"
  target_system_name: "armeabi"
  toolchain_identifier: "arm-linux-gnueabihf"

  tool_path { name: "ar" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-ar" }
  tool_path { name: "compat-ld" path: "/bin/false" }
  tool_path { name: "cpp" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-cpp" }
  tool_path { name: "dwp" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-dwp" }
  tool_path { name: "gcc" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-gcc" }
  tool_path { name: "gcov" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-gcov" }
  tool_path { name: "ld" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-ld" }

  tool_path { name: "nm" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-nm" }
  tool_path { name: "objcopy" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-objcopy" }
  tool_path { name: "objdump" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-objdump" }
  tool_path { name: "strip" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-strip" }

  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/arm-linux-gnueabihf/include/c++/4.9.3/"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/arm-linux-gnueabihf/sysroot/usr/include/"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/arm-linux-gnueabihf/libc/usr/include/"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/lib/gcc/arm-linux-gnueabihf/4.9.3/include"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/lib/gcc/arm-linux-gnueabihf/4.9.3/include-fixed"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/local_include"
  cxx_builtin_include_directory: "/usr/include"
  # The path below must match the one used in
  # tensorflow/tools/ci_build/pi/build_raspberry_pi.sh.
  cxx_builtin_include_directory: "/tmp/openblas_install/include/"
  cxx_flag: "-std=c++11"
  # The cxx_builtin_include_directory directives don't seem to be adding these, so
  # explicitly set them as flags. There's a query to the Bazel team outstanding about
  # why this is necessary.
  cxx_flag: "-isystem"
  cxx_flag: "/usr/include/arm-linux-gnueabihf"
  cxx_flag: "-isystem"
  cxx_flag: "%{PYTHON_INCLUDE_PATH}%"
  cxx_flag: "-isystem"
  cxx_flag: "/usr/include/"
  linker_flag: "-lstdc++"

  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""

  unfiltered_cxx_flag: "-no-canonical-prefixes"
  unfiltered_cxx_flag: "-fno-canonical-system-headers"

  compiler_flag: "-U_FORTIFY_SOURCE"
  compiler_flag: "-D_FORTIFY_SOURCE=1"
  compiler_flag: "-fstack-protector"
  compiler_flag: "-DRASPBERRY_PI"  # To differentiate from mobile builds.
  linker_flag: "-Wl,-z,relro,-z,now"

  linker_flag: "-no-canonical-prefixes"
  linker_flag: "-pass-exit-codes"

  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"

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
    linker_flag: "-Wl,--gc-sections"
  }
  linking_mode_flags { mode: DYNAMIC }

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
  target_libc: "local"
  target_cpu: "local"
  target_system_name: "local"
  toolchain_identifier: "local_linux"

  tool_path { name: "ar" path: "/usr/bin/ar" }
  tool_path { name: "compat-ld" path: "/usr/bin/ld" }
  tool_path { name: "cpp" path: "/usr/bin/cpp" }
  tool_path { name: "dwp" path: "/usr/bin/dwp" }
  tool_path { name: "gcc" path: "/usr/bin/gcc" }
  cxx_flag: "-std=c++0x"
  linker_flag: "-lstdc++"
  linker_flag: "-B/usr/bin/"

  # TODO(bazel-team): In theory, the path here ought to exactly match the path
  # used by gcc. That works because bazel currently doesn't track files at
  # absolute locations and has no remote execution, yet. However, this will need
  # to be fixed, maybe with auto-detection?
  cxx_builtin_include_directory: "/usr/lib/gcc/"
  cxx_builtin_include_directory: "/usr/local/include"
  cxx_builtin_include_directory: "/usr/include"
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
  unfiltered_cxx_flag: "-fno-canonical-system-headers"

  # Make C++ compilation deterministic. Use linkstamping instead of these
  # compiler symbols.
  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""

  # Security hardening on by default.
  # Conservative choice; -D_FORTIFY_SOURCE=2 may be unsafe in some cases.
  # We need to undef it before redefining it as some distributions now have
  # it enabled by default.
  compiler_flag: "-U_FORTIFY_SOURCE"
  compiler_flag: "-D_FORTIFY_SOURCE=1"
  compiler_flag: "-fstack-protector"
  linker_flag: "-Wl,-z,relro,-z,now"

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
  # Have gcc return the exit code from ld.
  linker_flag: "-pass-exit-codes"
  # Stamp the binary with a unique identifier.
  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"
  # Gold linker only? Can we enable this by default?
  # linker_flag: "-Wl,--warn-execstack"
  # linker_flag: "-Wl,--detect-odr-violations"

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
    linker_flag: "-Wl,--gc-sections"
  }
  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  abi_version: "local"
  abi_libc_version: "local"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "local"
  needsPic: true
  target_libc: "macosx"
  target_cpu: "darwin"
  target_system_name: "local"
  toolchain_identifier: "local_darwin"

  tool_path { name: "ar" path: "/usr/bin/libtool" }
  tool_path { name: "compat-ld" path: "/usr/bin/ld" }
  tool_path { name: "cpp" path: "/usr/bin/cpp" }
  tool_path { name: "dwp" path: "/usr/bin/dwp" }
  tool_path { name: "gcc" path: "osx_cc_wrapper.sh" }
  cxx_flag: "-std=c++0x"
  ar_flag: "-static"
  ar_flag: "-s"
  ar_flag: "-o"
  linker_flag: "-lstdc++"
  linker_flag: "-undefined"
  linker_flag: "dynamic_lookup"
  linker_flag: "-headerpad_max_install_names"
  # TODO(ulfjack): This is wrong on so many levels. Figure out a way to auto-detect the proper
  # setting from the local compiler, and also how to make incremental builds correct.
  cxx_builtin_include_directory: "/"
  tool_path { name: "gcov" path: "/usr/bin/gcov" }
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

  # Security hardening on by default.
  # Conservative choice; -D_FORTIFY_SOURCE=2 may be unsafe in some cases.
  compiler_flag: "-D_FORTIFY_SOURCE=1"
  compiler_flag: "-fstack-protector"

  # Enable coloring even if there's no attached terminal. Bazel removes the
  # escape sequences if --nocolor is specified.
  compiler_flag: "-fcolor-diagnostics"

  # All warnings are enabled. Maybe enable -Werror as well?
  compiler_flag: "-Wall"
  # Enable a few more warnings that aren't part of -Wall.
  compiler_flag: "-Wthread-safety"
  compiler_flag: "-Wself-assign"

  # Keep stack frames for debugging, even in opt mode.
  compiler_flag: "-fno-omit-frame-pointer"

  # Anticipated future default.
  linker_flag: "-no-canonical-prefixes"

  compilation_mode_flags {
    mode: DBG
    # Enable debug symbols.
    compiler_flag: "-g"
  }
  compilation_mode_flags {
    mode: OPT
    # No debug symbols.
    # Maybe we should enable https://gcc.gnu.org/wiki/DebugFission for opt or even generally?
    # However, that can't happen here, as it requires special handling in Bazel.
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
  }
  linking_mode_flags { mode: DYNAMIC }
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
  target_libc: "local"
  target_cpu: "freebsd"
  target_system_name: "local"
  toolchain_identifier: "local_freebsd"

  tool_path { name: "ar" path: "/usr/bin/ar" }
  tool_path { name: "compat-ld" path: "/usr/bin/ld" }
  tool_path { name: "cpp" path: "/usr/bin/cpp" }
  tool_path { name: "dwp" path: "/usr/bin/dwp" }
  tool_path { name: "gcc" path: "/usr/bin/clang" }
  cxx_flag: "-std=c++0x"
  linker_flag: "-lstdc++"
  linker_flag: "-B/usr/bin/"

  # TODO(bazel-team): In theory, the path here ought to exactly match the path
  # used by gcc. That works because bazel currently doesn't track files at
  # absolute locations and has no remote execution, yet. However, this will need
  # to be fixed, maybe with auto-detection?
  cxx_builtin_include_directory: "/usr/lib/clang"
  cxx_builtin_include_directory: "/usr/local/include"
  cxx_builtin_include_directory: "/usr/include"
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

  # Security hardening on by default.
  # Conservative choice; -D_FORTIFY_SOURCE=2 may be unsafe in some cases.
  # We need to undef it before redefining it as some distributions now have
  # it enabled by default.
  compiler_flag: "-U_FORTIFY_SOURCE"
  compiler_flag: "-D_FORTIFY_SOURCE=1"
  compiler_flag: "-fstack-protector"
  linker_flag: "-Wl,-z,relro,-z,now"

  # Enable coloring even if there's no attached terminal. Bazel removes the
  # escape sequences if --nocolor is specified. This isn't supported by gcc
  # on Ubuntu 14.04.
  # compiler_flag: "-fcolor-diagnostics"

  # All warnings are enabled. Maybe enable -Werror as well?
  compiler_flag: "-Wall"
  # Enable a few more warnings that aren't part of -Wall.
  #compiler_flag: "-Wunused-but-set-parameter"
  # But disable some that are problematic.
  #compiler_flag: "-Wno-free-nonheap-object" # has false positives

  # Keep stack frames for debugging, even in opt mode.
  compiler_flag: "-fno-omit-frame-pointer"

  # Anticipated future default.
  linker_flag: "-no-canonical-prefixes"
  # Have gcc return the exit code from ld.
  #linker_flag: "-pass-exit-codes"
  # Stamp the binary with a unique identifier.
  #linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"
  # Gold linker only? Can we enable this by default?
  # linker_flag: "-Wl,--warn-execstack"
  # linker_flag: "-Wl,--detect-odr-violations"

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
    linker_flag: "-Wl,--gc-sections"
  }
  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  abi_version: "local"
  abi_libc_version: "local"
  builtin_sysroot: ""
  compiler: "windows_mingw"
  host_system_name: "local"
  needsPic: false
  target_libc: "local"
  target_cpu: "x64_windows"
  target_system_name: "local"
  toolchain_identifier: "local_windows_mingw"

  tool_path { name: "ar" path: "C:/mingw/bin/ar" }
  tool_path { name: "compat-ld" path: "C:/mingw/bin/ld" }
  tool_path { name: "cpp" path: "C:/mingw/bin/cpp" }
  tool_path { name: "dwp" path: "C:/mingw/bin/dwp" }
  tool_path { name: "gcc" path: "C:/mingw/bin/gcc" }
  cxx_flag: "-std=c++0x"
  # TODO(bazel-team): In theory, the path here ought to exactly match the path
  # used by gcc. That works because bazel currently doesn't track files at
  # absolute locations and has no remote execution, yet. However, this will need
  # to be fixed, maybe with auto-detection?
  cxx_builtin_include_directory: "C:/mingw/include"
  cxx_builtin_include_directory: "C:/mingw/lib/gcc"
  tool_path { name: "gcov" path: "C:/mingw/bin/gcov" }
  tool_path { name: "ld" path: "C:/mingw/bin/ld" }
  tool_path { name: "nm" path: "C:/mingw/bin/nm" }
  tool_path { name: "objcopy" path: "C:/mingw/bin/objcopy" }
  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"
  tool_path { name: "objdump" path: "C:/mingw/bin/objdump" }
  tool_path { name: "strip" path: "C:/mingw/bin/strip" }
  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  abi_version: "local"
  abi_libc_version: "local"
  builtin_sysroot: ""
  compiler: "windows_msys64_mingw64"
  host_system_name: "local"
  needsPic: false
  target_libc: "local"
  target_cpu: "x64_windows"
  target_system_name: "local"
  toolchain_identifier: "local_windows_msys64_mingw64"

  tool_path { name: "ar" path: "C:/tools/msys64/mingw64/bin/ar" }
  tool_path { name: "compat-ld" path: "C:/tools/msys64/mingw64/bin/ld" }
  tool_path { name: "cpp" path: "C:/tools/msys64/mingw64/bin/cpp" }
  tool_path { name: "dwp" path: "C:/tools/msys64/mingw64/bin/dwp" }
  tool_path { name: "gcc" path: "C:/tools/msys64/mingw64/bin/gcc" }
  cxx_flag: "-std=c++0x"
  # TODO(bazel-team): In theory, the path here ought to exactly match the path
  # used by gcc. That works because bazel currently doesn't track files at
  # absolute locations and has no remote execution, yet. However, this will need
  # to be fixed, maybe with auto-detection?
  cxx_builtin_include_directory: "C:/tools/msys64/mingw64/x86_64-w64-mingw32/include"
  tool_path { name: "gcov" path: "C:/tools/msys64/mingw64/bin/gcov" }
  tool_path { name: "ld" path: "C:/tools/msys64/mingw64/bin/ld" }
  tool_path { name: "nm" path: "C:/tools/msys64/mingw64/bin/nm" }
  tool_path { name: "objcopy" path: "C:/tools/msys64/mingw64/bin/objcopy" }
  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"
  tool_path { name: "objdump" path: "C:/tools/msys64/mingw64/bin/objdump" }
  tool_path { name: "strip" path: "C:/tools/msys64/mingw64/bin/strip" }
  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  abi_version: "local"
  abi_libc_version: "local"
  builtin_sysroot: ""
  compiler: "windows_clang"
  host_system_name: "local"
  needsPic: false
  target_libc: "local"
  target_cpu: "x64_windows"
  target_system_name: "local"
  toolchain_identifier: "local_windows_clang"

  tool_path { name: "ar" path: "C:/mingw/bin/ar" }
  tool_path { name: "compat-ld" path: "C:/Program Files (x86)/LLVM/bin/ld" }
  tool_path { name: "cpp" path: "C:/Program Files (x86)/LLVM/bin/cpp" }
  tool_path { name: "dwp" path: "C:/Program Files (x86)/LLVM/bin/dwp" }
  tool_path { name: "gcc" path: "C:/Program Files (x86)/LLVM/bin/clang" }
  cxx_flag: "-std=c++0x"
  # TODO(bazel-team): In theory, the path here ought to exactly match the path
  # used by gcc. That works because bazel currently doesn't track files at
  # absolute locations and has no remote execution, yet. However, this will need
  # to be fixed, maybe with auto-detection?
  cxx_builtin_include_directory: "/usr/lib/gcc/"
  cxx_builtin_include_directory: "/usr/local/include"
  cxx_builtin_include_directory: "/usr/include"
  tool_path { name: "gcov" path: "C:/Program Files (x86)/LLVM/bin/gcov" }
  tool_path { name: "ld" path: "C:/Program Files (x86)/LLVM/bin/ld" }
  tool_path { name: "nm" path: "C:/Program Files (x86)/LLVM/bin/nm" }
  tool_path { name: "objcopy" path: "C:/Program Files (x86)/LLVM/bin/objcopy" }
  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"
  tool_path { name: "objdump" path: "C:/Program Files (x86)/LLVM/bin/objdump" }
  tool_path { name: "strip" path: "C:/Program Files (x86)/LLVM/bin/strip" }
  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
   abi_version: "local"
   abi_libc_version: "local"
   builtin_sysroot: ""
   compiler: "windows_msys64"
   host_system_name: "local"
   needsPic: false
   target_libc: "local"
   target_cpu: "x64_windows"
   target_system_name: "local"
   toolchain_identifier: "local_windows_msys64"

   tool_path { name: "ar" path: "C:/tools/msys64/usr/bin/ar" }
   tool_path { name: "compat-ld" path: "C:/tools/msys64/usr/bin/ld" }
   tool_path { name: "cpp" path: "C:/tools/msys64/usr/bin/cpp" }
   tool_path { name: "dwp" path: "C:/tools/msys64/usr/bin/dwp" }
   # Use gcc instead of g++ so that C will compile correctly.
   tool_path { name: "gcc" path: "C:/tools/msys64/usr/bin/gcc" }
   cxx_flag: "-std=gnu++0x"
   linker_flag: "-lstdc++"
   # TODO(bazel-team): In theory, the path here ought to exactly match the path
   # used by gcc. That works because bazel currently doesn't track files at
   # absolute locations and has no remote execution, yet. However, this will need
   # to be fixed, maybe with auto-detection?
   cxx_builtin_include_directory: "C:/tools/msys64/"
   cxx_builtin_include_directory: "/usr/"
   tool_path { name: "gcov" path: "C:/tools/msys64/usr/bin/gcov" }
   tool_path { name: "ld" path: "C:/tools/msys64/usr/bin/ld" }
   tool_path { name: "nm" path: "C:/tools/msys64/usr/bin/nm" }
   tool_path { name: "objcopy" path: "C:/tools/msys64/usr/bin/objcopy" }
   objcopy_embed_flag: "-I"
   objcopy_embed_flag: "binary"
   tool_path { name: "objdump" path: "C:/tools/msys64/usr/bin/objdump" }
   tool_path { name: "strip" path: "C:/tools/msys64/usr/bin/strip" }
   linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  toolchain_identifier: "vc_14_0_x64"
  host_system_name: "local"
  target_system_name: "local"

  abi_version: "local"
  abi_libc_version: "local"
  target_cpu: "x64_windows_msvc"
  compiler: "cl"
  target_libc: "msvcrt140"
  default_python_version: "python2.7"
  cxx_builtin_include_directory: "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/INCLUDE"
  cxx_builtin_include_directory: "C:/Program Files (x86)/Windows Kits/10/include/"
  cxx_builtin_include_directory: "C:/Program Files (x86)/Windows Kits/8.1/include/"
  cxx_builtin_include_directory: "C:/Program Files (x86)/GnuWin32/include/"
  cxx_builtin_include_directory: "C:/python_27_amd64/files/include"
  tool_path {
    name: "ar"
    path: "wrapper/bin/msvc_link.bat"
  }
  tool_path {
    name: "cpp"
    path: "wrapper/bin/msvc_cl.bat"
  }
  tool_path {
    name: "gcc"
    path: "wrapper/bin/msvc_cl.bat"
  }
  tool_path {
    name: "gcov"
    path: "wrapper/bin/msvc_nop.bat"
  }
  tool_path {
    name: "ld"
    path: "wrapper/bin/msvc_link.bat"
  }
  tool_path {
    name: "nm"
    path: "wrapper/bin/msvc_nop.bat"
  }
  tool_path {
    name: "objcopy"
    path: "wrapper/bin/msvc_nop.bat"
  }
  tool_path {
    name: "objdump"
    path: "wrapper/bin/msvc_nop.bat"
  }
  tool_path {
    name: "strip"
    path: "wrapper/bin/msvc_nop.bat"
  }
  supports_gold_linker: false
  supports_start_end_lib: false
  supports_interface_shared_objects: false
  supports_incremental_linker: false
  supports_normalizing_ar: true
  needsPic: false

  compiler_flag: "-m64"
  compiler_flag: "/D__inline__=__inline"
  # TODO(pcloudy): Review those flags below, they should be defined by cl.exe
  compiler_flag: "/DOS_WINDOWS=OS_WINDOWS"
  compiler_flag: "/DCOMPILER_MSVC"

  # Don't pollute with GDI macros in windows.h.
  compiler_flag: "/DNOGDI"
  # Don't define min/max macros in windows.h.
  compiler_flag: "/DNOMINMAX"
  compiler_flag: "/DPRAGMA_SUPPORTED"
  # Platform defines.
  compiler_flag: "/D_WIN32_WINNT=0x0600"
  # Turn off warning messages.
  compiler_flag: "/D_CRT_SECURE_NO_DEPRECATE"
  compiler_flag: "/D_CRT_SECURE_NO_WARNINGS"
  compiler_flag: "/D_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS"
  # Use math constants (M_PI, etc.) from the math library
  compiler_flag: "/D_USE_MATH_DEFINES"

  # Useful options to have on for compilation.
  # Suppress startup banner.
  compiler_flag: "/nologo"
  # Increase the capacity of object files to 2^32 sections.
  compiler_flag: "/bigobj"
  # Allocate 500MB for precomputed headers.
  compiler_flag: "/Zm500"
  # Use unsigned char by default.
  compiler_flag: "/J"
  # Use function level linking.
  compiler_flag: "/Gy"
  # Use string pooling.
  compiler_flag: "/GF"
  # Warning level 3 (could possibly go to 4 in the future).
  compiler_flag: "/W3"
  # Catch both asynchronous (structured) and synchronous (C++) exceptions.
  compiler_flag: "/EHsc"

  # Globally disabled warnings.
  # Don't warn about elements of array being be default initialized.
  compiler_flag: "/wd4351"
  # Don't warn about no matching delete found.
  compiler_flag: "/wd4291"
  # Don't warn about diamond inheritance patterns.
  compiler_flag: "/wd4250"
  # Don't warn about insecure functions (e.g. non _s functions).
  compiler_flag: "/wd4996"

  linker_flag: "-m64"

  feature {
    name: 'include_paths'
    flag_set {
      action: 'preprocess-assemble'
      action: 'c-compile'
      action: 'c++-compile'
      action: 'c++-header-parsing'
      action: 'c++-header-preprocessing'
      action: 'c++-module-compile'
      flag_group {
        flag: '/I%{quote_include_paths}'
      }
      flag_group {
        flag: '/I%{include_paths}'
      }
      flag_group {
        flag: '/I%{system_include_paths}'
      }
    }
  }

  feature {
    name: 'dependency_file'
    flag_set {
      action: 'assemble'
      action: 'preprocess-assemble'
      action: 'c-compile'
      action: 'c++-compile'
      action: 'c++-module-compile'
      action: 'c++-header-preprocessing'
      action: 'c++-header-parsing'
      expand_if_all_available: 'dependency_file'
      flag_group {
        flag: '/DEPENDENCY_FILE'
        flag: '%{dependency_file}'
      }
    }
  }

  # Stop passing -frandom-seed option
  feature {
    name: 'random_seed'
  }

  # This feature is just for enabling flag_set in action_config for -c and -o options during the transitional period
  feature {
    name: 'compile_action_flags_in_flag_set'
  }

  action_config {
    config_name: 'c-compile'
    action_name: 'c-compile'
    tool {
      tool_path: 'wrapper/bin/msvc_cl.bat'
    }
    flag_set {
      flag_group {
        flag: '/c'
        flag: '%{source_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_object_file'
      flag_group {
        flag: '/Fo%{output_object_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_assembly_file'
      flag_group {
        flag: '/Fa%{output_assembly_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_preprocess_file'
      flag_group {
        flag: '/P'
        flag: '/Fi%{output_preprocess_file}'
      }
    }
  }

  action_config {
    config_name: 'c++-compile'
    action_name: 'c++-compile'
    tool {
      tool_path: 'wrapper/bin/msvc_cl.bat'
    }
    flag_set {
      flag_group {
        flag: '/c'
        flag: '%{source_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_object_file'
      flag_group {
        flag: '/Fo%{output_object_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_assembly_file'
      flag_group {
        flag: '/Fa%{output_assembly_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_preprocess_file'
      flag_group {
        flag: '/P'
        flag: '/Fi%{output_preprocess_file}'
      }
    }
  }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "/DDEBUG=1"
    # This will signal the wrapper that we are doing a debug build, which sets
    # some internal state of the toolchain wrapper. It is intentionally a "-"
    # flag to make this very obvious.
    compiler_flag: "-g"
    compiler_flag: "/Od"
    compiler_flag: "-Xcompilation-mode=dbg"
  }

  compilation_mode_flags {
    mode: FASTBUILD
    compiler_flag: "/DNDEBUG"
    compiler_flag: "/Od"
    compiler_flag: "-Xcompilation-mode=fastbuild"
  }

  compilation_mode_flags {
    mode: OPT
    compiler_flag: "/DNDEBUG"
    compiler_flag: "/O2"
    compiler_flag: "-Xcompilation-mode=opt"
  }
}
