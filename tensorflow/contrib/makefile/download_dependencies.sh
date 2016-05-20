#!/bin/bash -x

DOWNLOADS_DIR=tensorflow/contrib/makefile/downloads

mkdir ${DOWNLOADS_DIR}

EIGEN_HASH=a5e9085a94e8
curl "https://bitbucket.org/eigen/eigen/get/${EIGEN_HASH}.tar.gz" \
-o /tmp/eigen-${EIGEN_HASH}.tar.gz
tar xzf /tmp/eigen-${EIGEN_HASH}.tar.gz -C ${DOWNLOADS_DIR}

git clone https://github.com/google/re2.git ${DOWNLOADS_DIR}/re2
git clone https://github.com/google/gemmlowp.git ${DOWNLOADS_DIR}/gemmlowp

# JPEG_VERSION=v9a
# curl "http://www.ijg.org/files/jpegsrc.${JPEG_VERSION}.tar.gz" \
# -o /tmp/jpegsrc.${JPEG_VERSION}.tar.gz
# tar xzf /tmp/jpegsrc.${JPEG_VERSION}.tar.gz -C ${DOWNLOADS_DIR}

# PNG_VERSION=v1.2.53
# curl -L "https://github.com/glennrp/libpng/archive/${PNG_VERSION}.zip" \
# -o /tmp/pngsrc.${PNG_VERSION}.zip
# unzip /tmp/pngsrc.${PNG_VERSION}.zip -d ${DOWNLOADS_DIR}
