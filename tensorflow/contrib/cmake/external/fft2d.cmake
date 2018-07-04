# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

include (ExternalProject)

set(fft2d_URL https://mirror.bazel.build/www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz)
set(fft2d_HASH SHA256=52bb637c70b971958ec79c9c8752b1df5ff0218a4db4510e60826e0cb79b5296)
set(fft2d_BUILD ${CMAKE_CURRENT_BINARY_DIR}/fft2d/)
set(fft2d_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/fft2d/src)

if(WIN32)
  set(fft2d_STATIC_LIBRARIES ${fft2d_BUILD}/src/lib/fft2d.lib)

  ExternalProject_Add(fft2d
      PREFIX fft2d
      URL ${fft2d_URL}
      URL_HASH ${fft2d_HASH}
      DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
      BUILD_IN_SOURCE 1
      BUILD_BYPRODUCTS ${fft2d_STATIC_LIBRARIES}
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/patches/fft2d/CMakeLists.txt ${fft2d_BUILD}/src/fft2d/CMakeLists.txt
      INSTALL_DIR ${fft2d_INSTALL}
      CMAKE_CACHE_ARGS
          -DCMAKE_BUILD_TYPE:STRING=Release
          -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
          -DCMAKE_INSTALL_PREFIX:STRING=${fft2d_INSTALL}
      GIT_SHALLOW 1
      GIT_PROGRESS 1
  )
else()
  set(fft2d_STATIC_LIBRARIES ${fft2d_BUILD}/src/fft2d/libfft2d.a)

  ExternalProject_Add(fft2d
      PREFIX fft2d
      URL ${fft2d_URL}
      URL_HASH ${fft2d_HASH}
      DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
      BUILD_IN_SOURCE 1
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/patches/fft2d/CMakeLists.txt ${fft2d_BUILD}/src/fft2d/CMakeLists.txt
      INSTALL_DIR ${fft2d_INSTALL}
      INSTALL_COMMAND echo
      BUILD_COMMAND $(MAKE)
      GIT_SHALLOW 1
      GIT_PROGRESS 1
  )

endif()
