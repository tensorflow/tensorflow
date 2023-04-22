# Copyright Northeastern University
# All Rights Reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Author: Nicolas Agostini
# Edited: Jude Haris

licenses(["notice"])

package(default_visibility = ["//visibility:public"])


genrule(
    name = "libsystemc",
    srcs = [],
    outs = ["systemc-2.3.3/install/lib/libsystemc.a"],
    cmd =
        "export CXX='g++ -fPIC' && export CC=gcc && " +
        "cmake -DCMAKE_INSTALL_PREFIX=external/systemc/systemc-2.3.3/install -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_INCLUDEDIR=include/systemc -DBUILD_SHARED_LIBS=off -Bexternal/systemc/systemc-2.3.3/build -Hexternal/systemc/systemc-2.3.3 &&" +
        "make -C external/systemc/systemc-2.3.3/build install -j4 CXXFLAGS='-fPIC' &&" +
        "cp external/systemc/systemc-2.3.3/install/lib/libsystemc.a $@",
)


cc_library(
    name = "systemc",
    srcs = ["systemc-2.3.3/install/lib/libsystemc.a"],
    hdrs = glob([
        "systemc-2.3.3/install/include/system.h",
    ]),
    copts = ["std=c++14"],
    data = [":libsystemc"],
    includes = [
        "systemc-2.3.3/install/include",
        "systemc-2.3.3/install/include/systemc",
    ],
)
