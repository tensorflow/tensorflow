RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    curl

# Install bazel
ARG BAZEL_VERSION=3.7.2
RUN mkdir /bazel && \
    curl -fSsL -o /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    curl -fSsL -o /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    bash /bazel/installer.sh && \
    rm -f /bazel/installer.sh
