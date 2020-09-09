FROM tensorflow/tensorflow:devel

ENV ANDROID_DEV_HOME /android
RUN mkdir -p ${ANDROID_DEV_HOME}

# Install Android SDK.
ENV ANDROID_SDK_FILENAME tools_r25.2.5-linux.zip
ENV ANDROID_SDK_URL https://dl.google.com/android/repository/${ANDROID_SDK_FILENAME}
ENV ANDROID_API_LEVEL 23
ENV ANDROID_NDK_API_LEVEL 18
# Build Tools Version liable to change.
ENV ANDROID_BUILD_TOOLS_VERSION 28.0.0
ENV ANDROID_SDK_HOME ${ANDROID_DEV_HOME}/sdk
ENV PATH ${PATH}:${ANDROID_SDK_HOME}/tools:${ANDROID_SDK_HOME}/platform-tools
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_SDK_URL} && \
    unzip ${ANDROID_SDK_FILENAME} -d android-sdk-linux && \
    rm ${ANDROID_SDK_FILENAME} && \
    bash -c "ln -s ${ANDROID_DEV_HOME}/android-sdk-* ${ANDROID_SDK_HOME}"

# Install Android NDK.
ENV ANDROID_NDK_FILENAME android-ndk-r17c-linux-x86_64.zip
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
