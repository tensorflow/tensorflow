licenses(["notice"])  # 3-Clause BSD

exports_files(["LICENSE"])

# Taken from: https://github.com/rnburn/satyr/blob/master/bazel/tbb.BUILD
# License for this BUILD file: MIT
# See: https://github.com/rnburn/satyr/blob/master/LICENSE
#
# License for TBB: Apache 2.0
# See: https://github.com/01org/tbb/blob/tbb_2018/LICENSE

genrule(
    name = "build_tbb",
    srcs = glob(["**"]) + [
        "@local_config_cc//:toolchain",
    ],
    outs = [
        "libtbb.a",
        "libtbbmalloc.a",
    ],
    cmd = """
	    set -e
	    WORK_DIR=$$PWD
		DEST_DIR=$$PWD/$(@D)
        export PATH=$$(dirname $(AR)):$$PATH
		export CXXFLAGS=$(CC_FLAGS)
		export NM=$(NM)
		export AR=$(AR)
		cd $$(dirname $(location :Makefile))

        #TBB's build needs some help to figure out what compiler it's using
        if $$CXX --version | grep clang &> /dev/null; then 
           COMPILER_OPT="compiler=clang"
        else
			COMPILER_OPT="compiler=gcc"

          #  # Workaround for TBB bug
          #  # See https://github.com/01org/tbb/issues/59
          #  CXXFLAGS="$$CXXFLAGS -flifetime-dse=1"
        fi 

        # uses extra_inc=big_iron.inc to specify that static libraries are
        # built. See https://software.intel.com/en-us/forums/intel-threading-building-blocks/topic/297792
        make tbb_build_prefix="build" \
              extra_inc=big_iron.inc \
              $$COMPILER_OPT; \

        echo cp build/build_{release,debug}/*.a $$DEST_DIR
        cp build/build_{release,debug}/*.a $$DEST_DIR
		cd $$WORK_DIR
	""",
)

cc_library(
    name = "tbb",
    srcs = ["libtbb.a"],
    hdrs = glob([
        "include/serial/**",
        "include/tbb/**/**",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
