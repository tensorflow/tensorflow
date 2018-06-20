FROM launcher.gcr.io/google/rbe-debian8:r327695
LABEL maintainer="Yu Yi <yiyu@google.com>"

# Copy install scripts
COPY install/*.sh /install/

# Setup envvars
ENV CC /usr/local/bin/clang
ENV CXX /usr/local/bin/clang++
ENV AR /usr/bin/ar

# Run pip install script for RBE Debian8 container.
RUN /install/install_pip_packages_remote.sh
RUN /install/install_pip_packages.sh
