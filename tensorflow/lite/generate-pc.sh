#!/usr/bin/env bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

TF_PREFIX='/usr/local'
LIBDIR='lib'

usage() {
    echo "Usage: $0 OPTIONS"
    echo -e "-p, --prefix\tset installation prefix (default: /usr/local)"
    echo -e "-l, --libdir\tset lib directory (default: lib)"
    echo -e "-v, --version\tset TensorFlow version"
    echo -e "-h, --help\tdisplay this message"
}

[ $# == 0 ] && usage && exit 0

# read the options
ARGS=$(getopt -o p:l:v:h --long prefix:,libdir:,version:,help -n $0 -- "$@")
eval set -- "$ARGS"

# extract options and their arguments into variables.
while true ; do
    case "$1" in
        -h|--help) usage ; exit ;;
        -p|--prefix)
            case "$2" in
                "") shift 2 ;;
                *) TF_PREFIX=$2 ; shift 2 ;;
            esac ;;
        -l|--libdir)
            case "$2" in
                "") shift 2 ;;
                *) LIBDIR=$2 ; shift 2 ;;
            esac ;;
        -v|--version)
            case "$2" in
                "") shift 2 ;;
                *) TF_VERSION=$2 ; shift 2 ;;
            esac ;;
        --) shift ; break ;;
        *) echo "Internal error! Try '$0 --help' for more information." ; exit 1 ;;
    esac
done

[ -z $TF_VERSION ] && echo "Specify a version using -v or --version" && exit 1

echo "Generating pkgconfig file for TensorFlowLite $TF_VERSION in $TF_PREFIX"

cat << EOF > tensorflowlite.pc
prefix=${TF_PREFIX}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/${LIBDIR}
includedir=\${prefix}/include/tensorflow

Name: TensorFlowLite
Version: ${TF_VERSION}
Description: Library for deploying models on mobile, microcontrollers and other edge devices.
Requires:
Libs: -L\${libdir} -ltensorflowlite
Cflags: -I\${includedir}
EOF
