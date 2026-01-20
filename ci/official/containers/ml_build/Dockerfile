################################################################################
ARG BASE_IMAGE=ubuntu:22.04@sha256:58b87898e82351c6cf9cf5b9f3c20257bb9e2dcf33af051e12ce532d7f94e3fe
FROM $BASE_IMAGE AS devel
# See https://docs.docker.com/reference/dockerfile/#understand-how-arg-and-from-interact
# on why we cannot reference BASE_IMAGE again unless we declare it again.
################################################################################

# Install devtoolset build dependencies
COPY setup.sources.sh /setup.sources.sh
COPY setup.packages.sh /setup.packages.sh
COPY builder.packages.txt /builder.packages.txt

RUN /setup.sources.sh && /setup.packages.sh /builder.packages.txt

# Setup Python
COPY setup.python.sh /setup.python.sh
COPY builder.requirements.txt /builder.requirements.txt
RUN /setup.python.sh python3.10 /builder.requirements.txt
RUN /setup.python.sh python3.11 /builder.requirements.txt
RUN /setup.python.sh python3.13 /builder.requirements.txt
RUN /setup.python.sh python3.13-nogil /builder.requirements.txt
RUN /setup.python.sh python3.14 /builder.requirements.txt
RUN /setup.python.sh python3.14-nogil /builder.requirements.txt

# Since we are using python3.12 as the default python version, we need to
# install python3.12 last for now.
# TODO(b/376338367): switch to pyenv.
RUN /setup.python.sh python3.12 /builder.requirements.txt

COPY setup.sources.cudnn.sh /setup.sources.cudnn.sh
ARG INSTALL_NVIDIA_PACKAGES=false
ARG NVIDIA_PACKAGES_FILE=cuda12.1_cudnn9.1.packages.txt
COPY ${NVIDIA_PACKAGES_FILE} /nvidia.packages.txt

RUN if [ "${INSTALL_NVIDIA_PACKAGES}" = "true" ]; then \
      echo "Installing Nvidia packages"; \
      /setup.sources.cudnn.sh && /setup.packages.sh /nvidia.packages.txt; \
    else \
      echo "Nvidia packages are not installed"; \
    fi

# Setup links for TensorFlow to compile.
# Referenced in devel.usertools/*.bazelrc.
# Set python3.12 as the default python version.
# TF does not support python3.13.
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3
RUN ln -sf /usr/bin/python3.12 /usr/bin/python
RUN ln -sf /usr/lib/python3.12 /usr/lib/tf_python

# Link the compat driver to the location if available.
RUN if [ -e "/usr/local/cuda/compat/libcuda.so.1" ]; then ln -s /usr/local/cuda/compat/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; fi

# Install various tools.
# - bats: bash unit testing framework
# - bazelisk: always use the correct bazel version
# - buildifier: clean bazel build deps
# - buildozer: clean bazel build deps
# - gcloud SDK: communicate with Google Cloud Platform (GCP) for RBE, CI
# - patchelf: Utility tool to modify existing ELF executables and libraries
RUN git clone --branch v1.13.0 https://github.com/bats-core/bats-core.git && bats-core/install.sh /usr/local && rm -rf bats-core
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64 -O /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel
RUN wget https://github.com/bazelbuild/buildtools/releases/download/v8.2.1/buildifier-linux-amd64 -O /usr/local/bin/buildifier && chmod +x /usr/local/bin/buildifier
RUN wget https://github.com/bazelbuild/buildtools/releases/download/v8.2.1/buildozer-linux-amd64 -O /usr/local/bin/buildozer && chmod +x /usr/local/bin/buildozer

RUN curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz | \
    tar zxf - google-cloud-sdk && \
    google-cloud-sdk/install.sh --quiet && \
    ln -s /google-cloud-sdk/bin/gcloud /usr/bin/gcloud
ENV PATH="$PATH:/google-cloud-sdk/bin/:/usr/local/cuda/bin/"

# Download and install patchelf v0.18.0 from GitHub. The default Ubuntu focal
# packages only provide the "0.10-2build1" version. We use patchelf to manipulate
# certain shared libraries during the wheel building process (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/build_pip_package.sh#L255-L262).
# When we use Patchelf versions <0.12, those shared libraries end up with a
# corrupted PT_NOTE program header. This was fixed in v0.12, see https://github.com/NixOS/patchelf/commit/43a33482b501b0f5ee9da312aabfca3806570cc9.
RUN wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz && tar -zxvf patchelf-0.18.0-x86_64.tar.gz -C /usr && rm -rf patchelf-0.18.0-x86_64.tar.gz

# Don't use the bazel cache when a new docker image is created.
RUN echo build --action_env=DOCKER_CACHEBUSTER=$(date +%s%N)$RANDOM >> /etc/bazel.bazelrc
RUN echo build --host_action_env=DOCKER_HOST_CACHEBUSTER=$(date +%s%N)$RANDOM >> /etc/bazel.bazelrc
