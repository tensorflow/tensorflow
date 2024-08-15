# Description:
#   libuv is a cross-platform asynchronous I/O library.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "uv",
    srcs = [
        "src/fs-poll.c",
        "src/idna.c",
        "src/inet.c",
        "src/random.c",
        "src/strscpy.c",
        "src/threadpool.c",
        "src/timer.c",
        "src/uv-common.c",
        "src/uv-data-getter-setters.c",
        "src/version.c",
    ] + [
        "src/unix/async.c",
        "src/unix/core.c",
        "src/unix/dl.c",
        "src/unix/fs.c",
        "src/unix/getaddrinfo.c",
        "src/unix/getnameinfo.c",
        "src/unix/loop.c",
        "src/unix/loop-watcher.c",
        "src/unix/pipe.c",
        "src/unix/poll.c",
        "src/unix/process.c",
        "src/unix/random-devurandom.c",
        "src/unix/signal.c",
        "src/unix/stream.c",
        "src/unix/tcp.c",
        "src/unix/thread.c",
        "src/unix/tty.c",
        "src/unix/udp.c",
    ] + select({
        "@platforms//os:osx": [
            "src/unix/bsd-ifaddrs.c",
            "src/unix/darwin.c",
            "src/unix/darwin-proctitle.c",
            "src/unix/fsevents.c",
            "src/unix/kqueue.c",
            "src/unix/proctitle.c",
            "src/unix/random-getentropy.c",
        ],
    }),
    # TODO: Add Linux, etc. as in https://github.com/libuv/libuv/blob/v1.38.0/CMakeLists.txt.
    hdrs = [
        "include/uv.h",
        "src/heap-inl.h",
        "src/idna.h",
        "src/queue.h",
        "src/strscpy.h",
        "src/unix/atomic-ops.h",
        "src/unix/internal.h",
        "src/unix/spinlock.h",
        "src/uv-common.h",
    ] + select({
        "@platforms//os:osx": [
            "src/unix/darwin-stub.h",
        ],
    }) + glob(["include/uv/*.h"]),
    copts = [
        "-fexceptions",
        "-Wno-unused-variable",
    ],
    includes = [
        "include",
        "src",
    ],
    textual_hdrs = [
        "include/uv.h",
    ],
)
