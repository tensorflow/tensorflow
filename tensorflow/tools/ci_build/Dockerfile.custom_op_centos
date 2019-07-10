FROM quay.io/aicoe/manylinux2010_x86_64:latest

LABEL maintainer="Amit Patankar <amitpatankar@google.com>"

# Copy and run the install scripts.
COPY install/*.sh /install/
RUN /install/install_yum_packages.sh

# Enable devtoolset-7 and python27 in the docker image.
env PATH="/opt/rh/python27/root/usr/bin:/opt/rh/devtoolset-7/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LD_LIBRARY_PATH="/opt/rh/python27/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:/opt/rh/devtoolset-7/root/usr/lib64/dyninst:/opt/rh/devtoolset-7/root/usr/lib/dyninst:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib" \
    PCP_DIR="/opt/rh/devtoolset-7/root" \
    PERL5LIB="/opt/rh/devtoolset-7/root//usr/lib64/perl5/vendor_perl:/opt/rh/devtoolset-7/root/usr/lib/perl5:/opt/rh/devtoolset-7/root//usr/share/perl5/vendor_perl" \
    PKG_CONFIG_PATH="/opt/rh/python27/root/usr/lib64/pkgconfig/"

RUN bash install/install_centos_python35.sh
RUN /install/install_centos_pip_packages.sh
RUN /install/install_bazel_from_source.sh
RUN /install/install_proto3.sh
RUN /install/install_buildifier.sh
RUN /install/install_golang.sh

# Set up the master bazelrc configuration file.
COPY install/.bazelrc /etc/bazel.bazelrc

