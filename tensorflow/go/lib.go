package tensorflow

// #cgo LDFLAGS: -L${SRCDIR}/../../bazel-bin/tensorflow -ltensorflow
// #cgo CFLAGS: -I${SRCDIR}/../../
// #cgo CXXFLAGS: -I${SRCDIR}/../../ -std=c++11 -stdlib=libc++
import "C"
