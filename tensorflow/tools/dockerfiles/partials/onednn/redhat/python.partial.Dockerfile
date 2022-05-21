# See http://bugs.python.org/issue19846
ARG PYTHON=python3
ARG PY_VER="38"

ARG PYTHON=python3

RUN INSTALL_PKGS="\
    python${PY_VER} \
    python${PY_VER}-pip \
    which" && \
    yum -y --setopt=tsflags=nodocs install $INSTALL_PKGS && \
    rpm -V $INSTALL_PKGS && \
    yum -y clean all --enablerepo='*'

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python
