# Options:
#   tensorflow-io
#   tensorflow-io-nightly
# Set --build-arg TF_IO_PACKAGE_VERSION=0.25.0 to install a specific version.
# Installs the latest version by default.
ARG TF_IO_PACKAGE=tensorflow-io
ARG TF_IO_PACKAGE_VERSION=
RUN python3 -m pip install --no-cache-dir ${TF_IO_PACKAGE}${TF_IO_PACKAGE_VERSION:+==${TF_IO_PACKAGE_VERSION}}
