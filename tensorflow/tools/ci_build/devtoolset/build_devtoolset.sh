#!/bin/bash -eu
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Builds a devtoolset cross-compiler targeting manylinux 2010 (glibc 2.12 /
# libstdc++ 4.4).

VERSION="$1"
TARGET="$2"

case "${VERSION}" in
devtoolset-7)
  LIBSTDCXX_VERSION="6.0.24"
  ;;
devtoolset-8)
  LIBSTDCXX_VERSION="6.0.25"
  ;;
*)
  echo "Usage: $0 {devtoolset-7|devtoolset-8} <target-directory>"
  exit 1
  ;;
esac

mkdir -p "${TARGET}"
# Download binary glibc 2.12 release.
wget "http://old-releases.ubuntu.com/ubuntu/pool/main/e/eglibc/libc6_2.12.1-0ubuntu6_amd64.deb" && \
    unar "libc6_2.12.1-0ubuntu6_amd64.deb" && \
    tar -C "${TARGET}" -xvzf "libc6_2.12.1-0ubuntu6_amd64/data.tar.gz" && \
    rm -rf "libc6_2.12.1-0ubuntu6_amd64.deb" "libc6_2.12.1-0ubuntu6_amd64"
wget "http://old-releases.ubuntu.com/ubuntu/pool/main/e/eglibc/libc6-dev_2.12.1-0ubuntu6_amd64.deb" && \
    unar "libc6-dev_2.12.1-0ubuntu6_amd64.deb" && \
    tar -C "${TARGET}" -xvzf "libc6-dev_2.12.1-0ubuntu6_amd64/data.tar.gz" && \
    rm -rf "libc6-dev_2.12.1-0ubuntu6_amd64.deb" "libc6-dev_2.12.1-0ubuntu6_amd64"

# Put the current kernel headers from ubuntu in place.
ln -s "/usr/include/linux" "/${TARGET}/usr/include/linux"
ln -s "/usr/include/asm-generic" "/${TARGET}/usr/include/asm-generic"
ln -s "/usr/include/x86_64-linux-gnu/asm" "/${TARGET}/usr/include/asm"

# Symlinks in the binary distribution are set up for installation in /usr, we
# need to fix up all the links to stay within /${TARGET}.
/fixlinks.sh "/${TARGET}"

# Patch to allow non-glibc 2.12 compatible builds to work.
sed -i '54i#define TCP_USER_TIMEOUT 18' "/${TARGET}/usr/include/netinet/tcp.h"

# Download binary libstdc++ 4.4 release we are going to link against.
# We only need the shared library, as we're going to develop against the
# libstdc++ provided by devtoolset.
wget "http://old-releases.ubuntu.com/ubuntu/pool/main/g/gcc-4.4/libstdc++6_4.4.3-4ubuntu5_amd64.deb" && \
    unar "libstdc++6_4.4.3-4ubuntu5_amd64.deb" && \
    tar -C "/${TARGET}" -xvzf "libstdc++6_4.4.3-4ubuntu5_amd64/data.tar.gz" "./usr/lib/libstdc++.so.6.0.13" && \
    rm -rf "libstdc++6_4.4.3-4ubuntu5_amd64.deb" "libstdc++6_4.4.3-4ubuntu5_amd64"

mkdir -p "${TARGET}-src"
cd "${TARGET}-src"

# Build a devtoolset cross-compiler based on our glibc 2.12 sysroot setup.

case "${VERSION}" in
devtoolset-7)
  wget "http://vault.centos.org/centos/6/sclo/Source/rh/devtoolset-7/devtoolset-7-gcc-7.3.1-5.15.el6.src.rpm"
  rpm2cpio "devtoolset-7-gcc-7.3.1-5.15.el6.src.rpm" |cpio -idmv
  tar -xvjf "gcc-7.3.1-20180303.tar.bz2" --strip 1
  ;;
devtoolset-8)
  wget "http://vault.centos.org/centos/6/sclo/Source/rh/devtoolset-8/devtoolset-8-gcc-8.2.1-3.el6.src.rpm"
  rpm2cpio "devtoolset-8-gcc-8.2.1-3.el6.src.rpm" |cpio -idmv
  tar -xvf "gcc-8.2.1-20180905.tar.xz" --strip 1
  ;;
esac

# Apply the devtoolset patches to gcc.
/rpm-patch.sh "gcc.spec"

./contrib/download_prerequisites

mkdir -p "${TARGET}-build"
cd "${TARGET}-build"

"${TARGET}-src/configure" \
      --prefix=/"${TARGET}/usr" \
      --with-sysroot="/${TARGET}" \
      --disable-bootstrap \
      --disable-libmpx \
      --disable-libsanitizer \
      --disable-libunwind-exceptions \
      --disable-libunwind-exceptions \
      --disable-lto \
      --disable-multilib \
      --enable-__cxa_atexit \
      --enable-gnu-indirect-function \
      --enable-gnu-unique-object \
      --enable-initfini-array \
      --enable-languages="c,c++" \
      --enable-linker-build-id \
      --enable-plugin \
      --enable-shared \
      --enable-threads=posix \
      --with-default-libstdcxx-abi="gcc4-compatible" \
      --with-gcc-major-version-only \
      --with-linker-hash-style="gnu" \
      --with-tune="generic" \
      && \
    make -j 42 && \
    make install

# Create the devtoolset libstdc++ linkerscript that links dynamically against
# the system libstdc++ 4.4 and provides all other symbols statically.
mv "/${TARGET}/usr/lib/libstdc++.so.${LIBSTDCXX_VERSION}" \
   "/${TARGET}/usr/lib/libstdc++.so.${LIBSTDCXX_VERSION}.backup"
echo -e "OUTPUT_FORMAT(elf64-x86-64)\nINPUT ( libstdc++.so.6.0.13 -lstdc++_nonshared44 )" \
   > "/${TARGET}/usr/lib/libstdc++.so.${LIBSTDCXX_VERSION}"
cp "./x86_64-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++_nonshared44.a" \
   "/${TARGET}/usr/lib"

# Link in architecture specific includes from the system; note that we cannot
# link in the whole x86_64-linux-gnu folder, as otherwise we're overlaying
# system gcc paths that we do not want to find.
# TODO(klimek): Automate linking in all non-gcc / non-kernel include
# directories.
mkdir -p "/${TARGET}/usr/include/x86_64-linux-gnu"
PYTHON_VERSIONS=("python3.6m" "python3.7m" "python3.8")
for v in "${PYTHON_VERSIONS[@]}"; do
  ln -s "/usr/local/include/${v}" "/${TARGET}/usr/include/x86_64-linux-gnu/${v}"
done

