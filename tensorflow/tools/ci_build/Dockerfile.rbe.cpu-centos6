# To push a new version, run:
# $ docker build -f Dockerfile.rbe.cpu-centos6 \
#       --tag "gcr.io/tensorflow-testing/nosla-centos6" .
# $ docker push gcr.io/tensorflow-testing/nosla-centos6

FROM quay.io/aicoe/manylinux2010_x86_64:latest
LABEL maintainer="Amit Patankar <amitpatankar@google.com>"

# Install packages required to build tensorflow.
RUN yum install -y centos-release-scl && \
    yum install -y \
      devtoolset-7 \
      java-1.8.0-openjdk-devel \
      patch \
      python27 \
      wget && \
    yum clean all -y

# Enable devtoolset-7 and python27 in the docker image.
env PATH="/opt/rh/python27/root/usr/bin:/opt/rh/devtoolset-7/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LD_LIBRARY_PATH="/opt/rh/python27/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:/opt/rh/devtoolset-7/root/usr/lib64/dyninst:/opt/rh/devtoolset-7/root/usr/lib/dyninst:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib" \
    PCP_DIR="/opt/rh/devtoolset-7/root" \
    PERL5LIB="/opt/rh/devtoolset-7/root//usr/lib64/perl5/vendor_perl:/opt/rh/devtoolset-7/root/usr/lib/perl5:/opt/rh/devtoolset-7/root//usr/share/perl5/vendor_perl" \
    PKG_CONFIG_PATH="/opt/rh/python27/root/usr/lib64/pkgconfig/"

# Install pip packages needed to build tensorflow.
COPY install/*.sh /install/
RUN bash install/install_yum_packages.sh
RUN bash install/install_centos_python36.sh
RUN bash install/install_centos_pip_packages.sh

# Install golang.
RUN bash install/install_golang_centos.sh
env GOROOT=/usr/local/go
env PATH=$GOROOT/bin:$PATH

# Install a /usr/bin/python2 and /usr/bin/python3 link.
# centos by default does not provide links, and instead relies on paths into
# /opt/ to switch to alternative configurations. For bazel remote builds,
# the python path between the local machine running bazel and the remote setup
# must be the same.
RUN update-alternatives --install /usr/bin/python2 python2 /opt/rh/python27/root/usr/bin/python2.7 0
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.6 0

# Install a ubuntu-compatible openjdk link so that ubuntu JAVA_HOME works
# for this image.
# TODO(klimek): Figure out a way to specify a different remote java path from
# the local one.
RUN ln -s /usr/lib/jvm/java /usr/lib/jvm/java-8-openjdk-amd64
