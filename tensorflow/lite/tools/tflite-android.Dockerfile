# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

FROM tensorflow/build:latest-python3.11

ENV ANDROID_DEV_HOME /android
RUN mkdir -p ${ANDROID_DEV_HOME}

RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk

# Install Android SDK.
ENV ANDROID_SDK_FILENAME commandlinetools-linux-6858069_latest.zip
ENV ANDROID_SDK_URL https://dl.google.com/android/repository/${ANDROID_SDK_FILENAME}
ENV ANDROID_API_LEVEL 30
ENV ANDROID_NDK_API_LEVEL 30
# Build Tools Version liable to change.
ENV ANDROID_BUILD_TOOLS_VERSION 31.0.0
ENV ANDROID_SDK_HOME ${ANDROID_DEV_HOME}/sdk
RUN mkdir -p ${ANDROID_SDK_HOME}/cmdline-tools
ENV PATH ${PATH}:${ANDROID_SDK_HOME}/cmdline-tools/latest/bin:${ANDROID_SDK_HOME}/platform-tools
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_SDK_URL} && \
    unzip ${ANDROID_SDK_FILENAME} -d /tmp && \
    mv /tmp/cmdline-tools ${ANDROID_SDK_HOME}/cmdline-tools/latest && \
    rm ${ANDROID_SDK_FILENAME}

# Install Android NDK.
ENV ANDROID_NDK_FILENAME android-ndk-r25b-linux.zip
ENV ANDROID_NDK_URL https://dl.google.com/android/repository/${ANDROID_NDK_FILENAME}
ENV ANDROID_NDK_HOME ${ANDROID_DEV_HOME}/ndk
ENV PATH ${PATH}:${ANDROID_NDK_HOME}
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_NDK_URL} && \
    unzip ${ANDROID_NDK_FILENAME} -d ${ANDROID_DEV_HOME} && \
    rm ${ANDROID_NDK_FILENAME} && \
    bash -c "ln -s ${ANDROID_DEV_HOME}/android-ndk-* ${ANDROID_NDK_HOME}"

# Make android ndk executable to all users.
RUN chmod -R go=u ${ANDROID_DEV_HOME}
