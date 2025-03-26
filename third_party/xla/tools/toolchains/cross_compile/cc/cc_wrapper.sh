#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# OS X relpath is not really working. This is a wrapper script around gcc
# to simulate relpath behavior.
#
# This wrapper uses install_name_tool to replace all paths in the binary
# (bazel-out/.../path/to/original/library.so) by the paths relative to
# the binary. It parses the command line to behave as rpath is supposed
# to work.
#
# See https://blogs.oracle.com/dipol/entry/dynamic_libraries_rpath_and_mac
# on how to set those paths for Mach-O binaries.
#
set -eu

LIBS=
LIB_DIRS=
RPATHS=
OUTPUT=

function parse_option() {
    local -r opt="$1"
    if [[ "${OUTPUT}" = "1" ]]; then
        OUTPUT=$opt
    elif [[ "$opt" =~ ^-l(.*)$ ]]; then
        LIBS="${BASH_REMATCH[1]} $LIBS"
    elif [[ "$opt" =~ ^-L(.*)$ ]]; then
        LIB_DIRS="${BASH_REMATCH[1]} $LIB_DIRS"
    elif [[ "$opt" =~ ^\@loader_path/(.*)$ ]]; then
        RPATHS="${BASH_REMATCH[1]} ${RPATHS}"
    elif [[ "$opt" = "-o" ]]; then
        # output is coming
        OUTPUT=1
    fi
}

# let parse the option list
for i in "$@"; do
    if [[ "$i" = @* && -r "${i:1}" ]]; then
        while IFS= read -r opt
        do
            parse_option "$opt"
        done < "${i:1}" || exit 1
    else
        parse_option "$i"
    fi
done

# Call the C++ compiler
/usr/lib/llvm-18/bin/clang "$@"

function get_library_path() {
    for libdir in ${LIB_DIRS}; do
        if [ -f ${libdir}/lib$1.so ]; then
            echo "${libdir}/lib$1.so"
        elif [ -f ${libdir}/lib$1.dylib ]; then
            echo "${libdir}/lib$1.dylib"
        fi
    done
}

# A convenient method to return the actual path even for non symlinks
# and multi-level symlinks, see b/300002682 for more details.
function get_realpath() {
    local mangled=$(echo $1 | sed 's/[-_\/a-zA-Z0-9]*_solib_darwin[-_a-zA-Z0-9]*\///g')
    if [[ "${mangled:0:3}" = "lib" ]]; then
        mangled="${mangled:3}"
    fi
    if [[ "${mangled:0:2}" = "_U" ]]; then
        mangled="${mangled:2}"
    fi
    local mangled_path=(${mangled//_S/ })
    local demangled_path=()
    for mangled in ${mangled_path[@]}; do
        demangled_path+=(${mangled//_U/_})
    done
    demangled_path=${demangled_path[@]}
    echo "bazel-out/darwin-opt/bin/${demangled_path// //}"
}

# Get the path of a lib inside a tool
function get_otool_path() {
    # the lib path is the path of the original lib relative to the workspace
    get_realpath $1 | sed 's|^.*/bazel-out/|bazel-out/|'
}

# Do replacements in the output
for rpath in ${RPATHS}; do
    for lib in ${LIBS}; do
        unset libname
        if [ -f "$(dirname ${OUTPUT})/${rpath}/lib${lib}.so" ]; then
            libname="lib${lib}.so"
        elif [ -f "$(dirname ${OUTPUT})/${rpath}/lib${lib}.dylib" ]; then
            libname="lib${lib}.dylib"
        fi
        # ${libname-} --> return $libname if defined, or undefined otherwise. This is to make
        # this set -e friendly
        if [[ -n "${libname-}" ]]; then
            libpath=$(get_library_path ${lib})
            if [ -n "${libpath}" ]; then
                /usr/lib/llvm-18/bin/llvm-install-name-tool -change $(get_otool_path "${libpath}") \
                    "@loader_path/${rpath}/${libname}" "${OUTPUT}"
            fi
        fi
    done
done