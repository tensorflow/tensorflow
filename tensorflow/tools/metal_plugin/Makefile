# Builds the Metal PluggableDevice plugin against an installed TensorFlow.
#
# The header and library paths come from the TensorFlow package itself rather
# than from a checked-in copy, so the plugin is built against exactly the
# TensorFlow it will be loaded into.

# Whichever interpreter has the TensorFlow you intend to load the plugin into.
# Override it when `python3` on PATH is not that one:
#   make PYTHON=/path/to/venv/bin/python
PYTHON ?= python3
# Overridable, because the build does not always run somewhere TensorFlow can
# be imported: pip isolates the build environment, so setup.py resolves these
# itself and passes them in.
TF_INCLUDE ?= $(shell $(PYTHON) -c "import tensorflow as tf; print(tf.sysconfig.get_include())" 2>/dev/null)
TF_LIB ?= $(shell $(PYTHON) -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" 2>/dev/null)

ifeq ($(strip $(TF_INCLUDE)),)
$(error TensorFlow was not found. Install it first, or pass TF_INCLUDE and \
TF_LIB explicitly)
endif
SDK := $(shell xcrun --sdk macosx --show-sdk-path 2>/dev/null)

BUILD := build
OUT := $(BUILD)/libmetal_plugin.dylib

SOURCES := src/plugin_init.cc \
           $(wildcard src/tensorflow/core/common_runtime/metal/*.mm) \
           $(wildcard src/tensorflow/core/common_runtime/metal/kernels/*.mm)
OBJECTS := $(patsubst src/%,$(BUILD)/%.o,$(SOURCES))

# The sources use manual retain and release, matching how TensorFlow's own
# objc_library targets compile them. Turning ARC on here would reject them.
#
# STREAM_OPTIONS probes the installed headers for a StreamExecutor C API
# callback added after the last release, so one source tree serves both an
# in-tree build and a build against whatever TensorFlow is installed.
STREAM_OPTIONS := $(shell bash tools/probe_stream_options.sh $(TF_INCLUDE))
ifeq ($(STREAM_OPTIONS),no)
COMPAT := -DTF_METAL_NO_STREAM_OPTIONS
endif

# -MMD -MP writes a .d file per object listing the headers it included, so a
# change to a header rebuilds everything that reads it. Without this, adding a
# field to a struct recompiled only the file it was declared in and left the
# rest reading the old layout, which shows up as a stream reporting a failure
# that never happened.
CXXFLAGS := -std=c++17 -O2 -fPIC -isysroot $(SDK) $(COMPAT) -MMD -MP \
            -mmacosx-version-min=13.0 \
            -Isrc -I$(TF_INCLUDE) \
            -I$(TF_INCLUDE)/external/farmhash_archive/src \
            -DNDEBUG -DTF_METAL_OUT_OF_TREE \
            -DTF_CAPI_WEAK

FRAMEWORKS := -framework Metal -framework MetalPerformanceShaders \
              -framework MetalPerformanceShadersGraph -framework Foundation

# Linking against the framework library rather than deferring every symbol to
# load time: an undefined symbol should be a build failure here, not a crash
# in someone's first import.
# TF_CAPI_WEAK, which TensorFlow's own c_api_macros.h provides for exactly this
# case, makes every C API reference a weak one. Part of the kernel C API is
# declared in the headers a released TensorFlow ships without being exported by
# any binary in it, and dyld on macOS 13 and later binds at load rather than at
# first call, so an ordinary reference to one of those would make dlopen fail
# outright. Weak references bind to null instead, and the kernels that need
# them are not registered when they are null (see ResourceVariableApiAvailable).
LDFLAGS := -dynamiclib $(FRAMEWORKS) \
           -L$(TF_LIB) -ltensorflow_framework.2 \
           -Wl,-undefined,dynamic_lookup \
           -Wl,-rpath,$(TF_LIB)

.PHONY: all clean test test-stream sweep install

all: $(OUT)

$(OUT): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "built $@"

$(BUILD)/%.o: src/%
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Fails loudly if the library still has unresolved TensorFlow or Metal symbols.
check-symbols: $(OUT)
	@bash tools/check_symbols.sh $(OUT)

test: $(OUT) test-stream
	$(PYTHON) tests/run_tests.py

# The StreamExecutor C API driven directly, without TensorFlow. Reaches what
# no op can: memset32 with a pattern that is not four equal bytes has one
# caller in the whole tree and it is a CUDA-only kernel.
$(BUILD)/stream_executor_test: tests/stream_executor_test.mm $(OUT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) tests/stream_executor_test.mm -o $@ \
	  -framework Foundation -L$(TF_LIB) -ltensorflow_framework.2 \
	  -Wl,-rpath,$(TF_LIB)

test-stream: $(BUILD)/stream_executor_test
	$(BUILD)/stream_executor_test $(OUT)

# Every registered op, through TensorFlow's own dispatch, against the CPU.
sweep: $(OUT)
	PYTHONPATH=tools $(PYTHON) tools/op_sweep.py

install: $(OUT)
	$(PYTHON) -m pip install .

clean:
	rm -rf $(BUILD)

-include $(OBJECTS:.o=.d)
