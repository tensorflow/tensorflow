major_version: "v1"
minor_version: "llvm:6.0.0"
default_target_cpu: "k8"

default_toolchain {
  cpu: "k8"
  toolchain_identifier: "k8-clang-6.0-cxx-4.8-linux-gnu"
}

toolchain {
  compiler: "clang6"         # bazel build --compiler=clang6
  target_cpu: "k8"           # bazel build --cpu=k8
  target_libc: "GLIBC_2.19"  # bazel build --glibc=GLIBC_2.19

  abi_libc_version: "2.19"
  abi_version: "gcc-4.8-cxx11"
  builtin_sysroot: ""
  cc_target_os: "linux-gnu"
  default_python_version: "python2.7"
  dynamic_runtimes_filegroup: "dynamic-runtime-libs-k8"
  host_system_name: "x86_64-unknown-linux-gnu"
  needsPic: true
  static_runtimes_filegroup: "static-runtime-libs-k8"
  supports_embedded_runtimes: true
  supports_fission: true
  supports_gold_linker: true
  supports_incremental_linker: true
  supports_interface_shared_objects: true
  supports_normalizing_ar: true
  supports_start_end_lib: true
  supports_thin_archives: true
  target_system_name: "x86_64-unknown-linux-gnu"
  toolchain_identifier: "k8-clang-6.0-cxx-4.8-linux-gnu"

  tool_path { name: "ar" path: "%package(@local_config_clang6//clang6)%/llvm/bin/llvm-ar" }
  tool_path { name: "as" path: "%package(@local_config_clang6//clang6)%/llvm/bin/llvm-as" }
  tool_path { name: "compat-ld" path: "%package(@local_config_clang6//clang6)%/llvm/bin/ld.lld" }
  tool_path { name: "cpp" path: "%package(@local_config_clang6//clang6)%/llvm/bin/llvm-cpp" }
  tool_path { name: "dwp" path: "%package(@local_config_clang6//clang6)%/llvm/bin/llvm-dwp" }
  tool_path { name: "gcc" path: "%package(@local_config_clang6//clang6)%/llvm/bin/clang" }
  tool_path { name: "gcov" path: "%package(@local_config_clang6//clang6)%/llvm/bin/llvm-cov" }
  tool_path { name: "ld" path: "%package(@local_config_clang6//clang6)%/llvm/bin/ld.lld" }
  tool_path { name: "llvm-profdata" path: "%package(@local_config_clang6//clang6)%/llvm/bin/llvm-profdata" }
  tool_path { name: "nm" path: "%package(@local_config_clang6//clang6)%/llvm/bin/llvm-nm" }
  tool_path { name: "objcopy" path: "%package(@local_config_clang6//clang6)%/llvm/bin/llvm-objcopy" }
  tool_path { name: "objdump" path: "%package(@local_config_clang6//clang6)%/sbin/objdump" }
  tool_path { name: "strip" path: "%package(@local_config_clang6//clang6)%/sbin/strip" }

  unfiltered_cxx_flag: "-no-canonical-prefixes"

  # Make C++ compilation deterministic. Use linkstamping instead of these
  # compiler symbols.
  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""

  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"

  # This action_config makes features flags propagate
  # to CC_FLAGS for genrules, and eventually skylark.
  action_config {
    action_name: "cc-flags-make-variable"
    config_name: "cc-flags-make-variable"
  }

  # Security hardening on by default.
  # Conservative choice; -D_FORTIFY_SOURCE=2 may be unsafe in some cases.
  # We need to undef it before redefining it as some distributions now have
  # it enabled by default.
  compiler_flag: "-U_FORTIFY_SOURCE"
  compiler_flag: "-D_FORTIFY_SOURCE=1"
  compiler_flag: "-fstack-protector"
  linker_flag: "-Wl,-z,relro,-z,now"

  # This adds a little bit more durability to our Clang build.
  #
  # At the moment, this only only be needed for:
  # - add_boringssl_s390x.patch: --Wa,--noexecstack
  #
  # Folks who do maintenance work on TF Bazel Clang should consider
  # commenting out these lines, while doing that work, to gain a better
  # understanding of what the intersection of support looks like between GCC
  # and Clang. Please note that,  Bazel does not support
  # -Xclang-only / -Xgcc-only.
  compiler_flag: "-Wno-unknown-warning-option"
  compiler_flag: "-Wno-unused-command-line-argument"
  compiler_flag: "-Wno-ignored-optimization-argument"

  #### Common compiler options. ####
  compiler_flag: "-D_REENTRANT"
  compiler_flag: "-D__STDC_FORMAT_MACROS"
  compiler_flag: "-DSUPPRESS_USE_FILE_OFFSET64"
  compiler_flag: "-Wall"
  compiler_flag: "-Wformat-security"
  compiler_flag: "-Wframe-larger-than=16384"
  compiler_flag: "-Wno-char-subscripts"
  compiler_flag: "-Wno-error=deprecated-declarations"
  compiler_flag: "-Wno-uninitialized"
  compiler_flag: "-Wno-sign-compare"
  compiler_flag: "-Wno-strict-overflow"
  compiler_flag: "-Wno-unused-function"
  compiler_flag: "-fdiagnostics-show-option"
  compiler_flag: "-fmessage-length=0"
  compiler_flag: "-fno-exceptions"
  compiler_flag: "-fno-omit-frame-pointer"
  compiler_flag: "-fno-strict-aliasing"
  compiler_flag: "-fno-use-init-array"
  compiler_flag: "-funsigned-char"
  compiler_flag: "-gmlt"
  cxx_flag: "-Wno-deprecated"
  cxx_flag: "-Wno-invalid-offsetof"  # Needed for protobuf code (2017-11-07)
  cxx_flag: "-fshow-overloads=best"
  compiler_flag: "-Wthread-safety-analysis"

  # Python extensions unfortunately make this go wild.
  compiler_flag: "-Wno-writable-strings"

  # GCC's warning produces too many false positives:
  cxx_flag: "-Woverloaded-virtual"
  cxx_flag: "-Wnon-virtual-dtor"

  # Enable coloring even if there's no attached terminal. Bazel removes the
  # escape sequences if --nocolor is specified. This isn't supported by gcc
  # on Ubuntu 14.04.
  compiler_flag: "-fcolor-diagnostics"

  # Disable some broken warnings from Clang.
  compiler_flag: "-Wno-ambiguous-member-template"
  compiler_flag: "-Wno-pointer-sign"

  # These warnings have a low signal to noise ratio.
  compiler_flag: "-Wno-reserved-user-defined-literal"
  compiler_flag: "-Wno-return-type-c-linkage"
  compiler_flag: "-Wno-invalid-source-encoding"

  # Per default we switch off any layering related warnings.
  compiler_flag: "-Wno-private-header"

  # Clang-specific warnings that we explicitly enable for TensorFlow. Some of
  # these aren't on by default, or under -Wall, or are subsets of warnings
  # turned off above.
  compiler_flag: "-Wfloat-overflow-conversion"
  compiler_flag: "-Wfloat-zero-conversion"
  compiler_flag: "-Wfor-loop-analysis"
  compiler_flag: "-Wgnu-redeclared-enum"
  compiler_flag: "-Winfinite-recursion"
  compiler_flag: "-Wliteral-conversion"
  compiler_flag: "-Wself-assign"
  compiler_flag: "-Wstring-conversion"
  compiler_flag: "-Wtautological-overlap-compare"
  compiler_flag: "-Wunused-comparison"
  compiler_flag: "-Wvla"
  cxx_flag: "-Wdeprecated-increment-bool"

  # Clang code-generation flags for performance optimization.
  compiler_flag: "-faligned-allocation"
  compiler_flag: "-fnew-alignment=8"

  # Clang defaults to C99 while GCC defaults to C89. GCC plugins are written in
  # C89 and don't have a BUILD rule we could add a copts flag to.
  gcc_plugin_compiler_flag: "-std=gnu89"

  compilation_mode_flags {
    mode: FASTBUILD
  }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "-g"
  }

  compilation_mode_flags {
    mode: OPT
    compiler_flag: "-g0"
    compiler_flag: "-fdebug-types-section"
    compiler_flag: "-DNDEBUG"
    compiler_flag: "-fno-split-dwarf-inlining"
    compiler_flag: "-Os"
    compiler_flag: "-fexperimental-new-pass-manager"
    compiler_flag: "-fdebug-info-for-profiling"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
    linker_flag: "-Wl,-z,relro,-z,now"
  }

  # Features indicating whether this is a host compile or not. Exactly one of
  # these will be implicitly provided by bazel.
  feature { name: "host" }
  feature { name: "nonhost" }

  # Features indicating which compiler will be used for code generation.
  feature {
    name: "llvm_codegen"
    provides: "codegen"
    enabled: true
  }

  # Features for compilation modes. Exactly one of these will be implicitly
  # provided by bazel.
  feature { name: "fastbuild" }
  feature { name: "dbg" }
  feature { name: "opt" }

  # Features controlling the C++ language mode.
  feature {
    name: "c++11"
    provides: "c++std"
    flag_set {
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      flag_group {
        flag: "-nostdinc++"
        flag: "-std=c++11"
        flag: "-Wc++14-extensions"
        flag: "-Wc++2a-extensions"
        flag: "-Wno-binary-literal"
      }
    }
  }
  feature {
    name: "c++14"
    provides: "c++std"
    flag_set {
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      flag_group {
        flag: "-nostdinc++"
        flag: "-std=c++14"
        flag: "-Wc++11-compat"
        flag: "-Wno-c++11-compat-binary-literal"
        flag: "-Wc++2a-extensions"
      }
    }
  }
  feature {
    name: "c++17"
    provides: "c++std"
    flag_set {
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      flag_group {
        flag: "-nostdinc++"
        flag: "-std=c++17"
        flag: "-Wc++11-compat"
        flag: "-Wno-c++11-compat-binary-literal"
        flag: "-Wc++2a-extensions"
      }
    }
  }
  feature {
    name: "c++2a"
    provides: "c++std"
    flag_set {
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      flag_group {
        flag: "-nostdinc++"
        flag: "-std=c++2a"
        flag: "-Wc++11-compat"
        flag: "-Wno-c++11-compat-binary-literal"
      }
    }
  }
  feature {
    name: "c++default"
    enabled: true
    flag_set {
      # Provide the c++11 flags if no standard is selected
      with_feature {
        not_feature: "c++11"
        not_feature: "c++14"
        not_feature: "c++17"
        not_feature: "c++2a"
      }
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      flag_group {
        flag: "-nostdinc++"
        flag: "-std=c++11"
        flag: "-Wc++14-extensions"
        flag: "-Wc++2a-extensions"
        flag: "-Wno-binary-literal"
      }
    }
  }

  feature {
    name: "use_compiler_rt"
    requires { feature: "llvm_codegen" }
    # TODO(saugustine): At the moment, "use_compiler_rt" also
    # requires "linking_mode_flags { mode: FULLY_STATIC" ... },
    # but that isn't a feature. We should probably convert it.
    flag_set {
      action: "c++-link"
      action: "c++-link-interface-dynamic-library"
      action: "c++-link-dynamic-library"
      action: "c++-link-executable"
      # "link" is a misnomer for these actions. They are really just
      # invocations of ar.
      #action: "c++-link-pic-static-library"
      #action: "c++-link-static-library"
      #action: "c++-link-alwayslink-static-library"
      #action: "c++-link-pic-static-library"
      #action: "c++-link-alwayslink-pic-static-library"
      flag_group {
        flag: "-rtlib=compiler-rt"
        flag: "-lunwind"
      }
    }
  }

  feature {
    name: "pie"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "cc-flags-make-variable"
      action: "lto-backend"
      action: "linkstamp-compile"
      flag_group {
        flag: "-mpie-copy-relocations"
        flag: "-fPIE"
      }
    }
    flag_set {
      action: "cc-flags-make-variable"
      action: "c++-link-executable"
      flag_group {
        flag: "-pie"
      }
    }
  }

  # Pic must appear after pie, because pic may need to override pie, and bazel
  # turns it on selectively. These don't interact with other options.
  #
  # TODO: In practice, normal vs pic vs pie is a ternary mode. We should
  # implement it that way. This will require changes to bazel, which only
  # calculates whether or not pic is needed, not pie.
  #
  # NOTE: Bazel might make this all a moot point.
  feature {
    name: "pic"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      expand_if_all_available: "pic"
      flag_group {
        flag: "-fPIC"
      }
    }
  }

  feature {
    name: "gold"
    enabled: true
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-interface-dynamic-library"
      flag_group {
        expand_if_none_available: "lto"
        flag: "-fuse-ld=gold"
      }
    }
  }

  # This is great if you want linking TensorFlow to take ten minutes.
  feature {
    name: "lto"
    requires { feature: "nonhost" }
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-flto=thin"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-interface-dynamic-library"
      flag_group {
        flag: "-flto=thin"
      }
    }
  }

  feature {
    name: "parse_headers"
    flag_set {
      action: "c++-header-parsing"
      flag_group {
        flag: "-xc++-header"
        flag: "-fsyntax-only"
      }
    }
  }

  feature {
    name: "preprocess_headers"
    flag_set {
      action: "c++-header-preprocessing"
      flag_group {
        flag: "-xc++"
        flag: "-E"
      }
    }
  }

  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      action: "lto-backend"
      flag_group {
        flag: "-gsplit-dwarf"
        flag: "-ggnu-pubnames"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-interface-dynamic-library"
      flag_group {
        expand_if_all_available: "is_using_fission"
        flag: "-Wl,--gdb-index"
      }
    }
  }

  feature {
    name: "xray"
    requires {
      feature: "llvm_codegen"
      feature: "nonhost"
    }
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "c++-module-compile"
      action: "c++-link-interface-dynamic-library"
      action: "c++-link-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fxray-instrument"
      }
    }
  }

  feature {
    name: "minimal_ubsan"
    requires { feature: "llvm_codegen" }
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      flag_group {
        flag: "-fsanitize=return,returns-nonnull-attribute,vla-bound,unreachable,float-cast-overflow"
        flag: "-fsanitize-trap=all"
        flag: "-DUNDEFINED_BEHAVIOR_SANITIZER"
      }
    }
  }

  feature {
    name: "minimal_ubsan_enabled_by_default"
    requires {
      feature: "llvm_codegen"
      feature: "fastbuild"
    }
    enabled: true
    implies: "minimal_ubsan"
  }

  cxx_builtin_include_directory: "%package(@local_config_clang6//clang6)%/llvm/lib/clang/6.0.0/include"
  cxx_builtin_include_directory: "/usr/include"

  unfiltered_cxx_flag: "-cxx-isystem"
  unfiltered_cxx_flag: "/usr/include/c++/4.8"
  unfiltered_cxx_flag: "-cxx-isystem"
  unfiltered_cxx_flag: "/usr/include/x86_64-linux-gnu/c++/4.8"
  unfiltered_cxx_flag: "-isystem"
  unfiltered_cxx_flag: "%package(@local_config_clang6//clang6)%/llvm/lib/clang/6.0.0/include"
  unfiltered_cxx_flag: "-isystem"
  unfiltered_cxx_flag: "/usr/include/x86_64-linux-gnu"
  unfiltered_cxx_flag: "-isystem"
  unfiltered_cxx_flag: "/usr/include"

  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--fatal-warnings"
  linker_flag: "-Wl,--hash-style=gnu"
  linker_flag: "-no-canonical-prefixes"
  linker_flag: "--target=x86_64-unknown-linux-gnu"

  linker_flag: "-L/usr/lib/gcc/x86_64-linux-gnu/4.8"

  # This is the minimum x86 architecture TensorFlow supports.
  compiler_flag: "-DARCH_K8"
  compiler_flag: "-m64"

  # These are for Linux.
  ld_embed_flag: "-melf_x86_64"
  linker_flag: "-Wl,--eh-frame-hdr"
  linker_flag: "-Wl,-z,max-page-size=0x1000"

  # Google never uses the stack like a heap, e.g. alloca(), because tcmalloc
  # and jemalloc are so fast. However copts=["$(STACK_FRAME_UNLIMITED)"] can be
  # specified when that can't be the case.
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }

  # These flags are for folks who build C/C++ code inside genrules.
  make_variable {
    name: "CC_FLAGS"
    value: "-no-canonical-prefixes --target=x86_64-unknown-linux-gnu -fno-omit-frame-pointer -fno-tree-vrp -msse3"
  }

  feature {
    name: "copts"
    flag_set {
      expand_if_all_available: "copts"
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      flag_group {
        iterate_over: "copts"
        flag: "%{copts}"
      }
    }
  }

  # Please do not statically link libstdc++. This would probably lead to a lot
  # of bloat since OpKernels need to use linkstatic=1 because  b/27630669 and
  # it could cause memory leaks since Python uses dlopen() on our libraries:
  # https://stackoverflow.com/a/35015415
  linker_flag: "-lstdc++"
  linker_flag: "-lm"
  linker_flag: "-lpthread"
  linker_flag: "-l:/lib/x86_64-linux-gnu/libc-2.19.so"
}
