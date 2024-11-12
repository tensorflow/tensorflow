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
# Builds a devtoolset cross-compiler targeting manylinux2014 (glibc 2.17 / libstdc++ 4.8).

VERSION="$1"
TARGET="$2"

case "${VERSION}" in
devtoolset-9)
  LIBSTDCXX_VERSION="6.0.28"
  LIBSTDCXX_ABI="new"
  ;;
devtoolset-10)
  LIBSTDCXX_VERSION="6.0.28"
  LIBSTDCXX_ABI="new"
  ;;
*)
  echo "Usage: $0 {devtoolset-9|devtoolset-10} <target-directory> <arch>"
  echo "Use 'devtoolset-9' to build a manylinux2014 compatible toolchain"
  exit 1
  ;;
esac

mkdir -p "${TARGET}"

mkdir -p ${TARGET}/usr/include

# Put the current kernel headers from ubuntu in place.
ln -s "/usr/include/linux" "${TARGET}/usr/include/linux"
ln -s "/usr/include/asm-generic" "${TARGET}/usr/include/asm-generic"
ln -s "/usr/include/aarch64-linux-gnu/asm" "${TARGET}/usr/include/asm"

# Download glibc's shared and development libraries based on the value of the
# `VERSION` parameter.
# Note: 'Templatizing' this and the other conditional branches would require
# defining several variables (version, os, path) making it difficult to maintain
# and extend for future modifications.
mkdir -p glibc-src
mkdir -p glibc-build
cd glibc-src
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --tries=5 "https://vault.centos.org/centos/7/os/Source/SPackages/glibc-2.17-317.el7.src.rpm"
rpm2cpio "glibc-2.17-317.el7.src.rpm" |cpio -idmv
tar -xvzf "glibc-2.17-c758a686.tar.gz" --strip 1
tar -xvzf "glibc-2.17-c758a686-releng.tar.gz" --strip 1
sed -i '/patch0060/d' glibc.spec
/rpm-patch.sh "glibc.spec"
rm -f "glibc-2.17-317.el7.src.rpm" "glibc-2.17-c758a686.tar.gz" "glibc-2.17-c758a686-releng.tar.gz"
patch -p1 < /gcc9-fixups.patch
patch -p1 < /stringop_trunc.patch
cd ../glibc-build
../glibc-src/configure --prefix=/usr --disable-werror --enable-obsolete-rpc --disable-profile
make -j$(nproc)
make install DESTDIR=${TARGET}
cd ..

# Symlinks in the binary distribution are set up for installation in /usr, we
# need to fix up all the links to stay within /${TARGET}.
/fixlinks.sh "/${TARGET}"

# Patch to allow non-glibc 2.12 compatible builds to work.
sed -i '54i#define TCP_USER_TIMEOUT 18' "/${TARGET}/usr/include/netinet/tcp.h"

# Download specific version of libstdc++ shared library based on the value of
# the `VERSION` parameter
  # Download binary libstdc++ 4.8 shared library release
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --tries=5 "http://old-releases.ubuntu.com/ubuntu/pool/main/g/gcc-4.8/libstdc++6_4.8.1-10ubuntu8_arm64.deb" && \
    unar "libstdc++6_4.8.1-10ubuntu8_arm64.deb" && \
    tar -C "${TARGET}" -xvzf "libstdc++6_4.8.1-10ubuntu8_arm64/data.tar.gz" "./usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.18"  && \
    rm -rf "libstdc++6_4.8.1-10ubuntu8_arm64.deb" "libstdc++6_4.8.1-10ubuntu8_arm64"

mkdir -p "${TARGET}-src"
cd "${TARGET}-src"

# Build a devtoolset cross-compiler based on our glibc 2.12/glibc 2.17 sysroot setup.
case "${VERSION}" in
devtoolset-9)
  wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --tries=5 "https://vault.centos.org/centos/7/sclo/Source/rh/devtoolset-9-gcc-9.3.1-2.2.el7.src.rpm"
  rpm2cpio "devtoolset-9-gcc-9.3.1-2.2.el7.src.rpm" |cpio -idmv
  tar -xvf "gcc-9.3.1-20200408.tar.xz" --strip 1
  ;;
devtoolset-10)
  wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --tries=5 "https://vault.centos.org/centos/7/sclo/Source/rh/devtoolset-10-gcc-10.2.1-11.2.el7.src.rpm"
  rpm2cpio "devtoolset-10-gcc-10.2.1-11.2.el7.src.rpm" |cpio -idmv
  tar -xvf "gcc-10.2.1-20210130.tar.xz" --strip 1
  ;;
esac

# Apply the devtoolset patches to gcc.
/rpm-patch.sh "gcc.spec"

./contrib/download_prerequisites

mkdir -p "${TARGET}-build"
cd "${TARGET}-build"

"${TARGET}-src/configure" \
      --prefix="${TARGET}/usr" \
      --with-sysroot="/${TARGET}" \
      --disable-bootstrap \
      --disable-libmpx \
      --enable-libsanitizer \
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
      --with-default-libstdcxx-abi=${LIBSTDCXX_ABI} \
      --with-gcc-major-version-only \
      --with-linker-hash-style="gnu" \
      && \
      make -j$(nproc) && \
      make install


# Create the devtoolset libstdc++ linkerscript that links dynamically against
# the system libstdc++ 4.4 and provides all other symbols statically.
# Note that the installation path for libstdc++ here is ${TARGET}/usr/lib64/
mv "${TARGET}/usr/lib64/libstdc++.so.${LIBSTDCXX_VERSION}" \
   "${TARGET}/usr/lib64/libstdc++.so.${LIBSTDCXX_VERSION}.backup"
echo -e "OUTPUT_FORMAT(elf64-littleaarch64)\nINPUT ( libstdc++.so.6.0.18 -lstdc++_nonshared44 )" \
   > "${TARGET}/usr/lib64/libstdc++.so.${LIBSTDCXX_VERSION}"
cp "./aarch64-unknown-linux-gnu/libstdc++-v3/src/.libs/libstdc++_nonshared44.a" \
   "${TARGET}/usr/lib64"


# Link in architecture specific includes from the system; note that we cannot
# link in the whole aarch64-linux-gnu folder, as otherwise we're overlaying
# system gcc paths that we do not want to find.
# TODO(klimek): Automate linking in all non-gcc / non-kernel include
# directories.
mkdir -p "${TARGET}/usr/include/aarch64-linux-gnu"
PYTHON_VERSIONS=("python3.9" "python3.10" "python3.11" "python3.12")
for v in "${PYTHON_VERSIONS[@]}"; do
  ln -s "/usr/local/include/${v}" "${TARGET}/usr/include/aarch64-linux-gnu/${v}"
done
