CFLAGS := -I../..
# LDFLAGS := -L/usr/local/lib/tensorflow
# LDFLAGS := -L../../bazel-bin/tensorflow
LDFLAGS := -L../../bazel-out/local-fastbuild/bin/tensorflow

SFLAGS=-Xcc $(CFLAGS) -Xlinker $(LDFLAGS)

DBGLIB=.build/debug/TensorFlow.swiftmodule
RELLIB=.build/release/TensorFlow.swiftmodule

all: $(DBGLIB)

$(DBGLIB): Sources/*
	swift build $(SFLAGS)

release: $(RELLIB)

$(RELLIB): Sources/*
	swift build -c release $(SFLAGS)

test: $(DBGLIB)
	swift test $(SFLAGS)

clean:
	rm -rf $(DBGLIB) $(RELLIB)
	swift build --clean
