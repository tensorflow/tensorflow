# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PYTHON=python3

RUN yum --disableplugin=subscription-managerupdate -y && yum --disableplugin=subscription-managerinstall -y \
    ${PYTHON} \
    ${PYTHON}-pip \
    which && \
    yum --disableplugin=subscription-managerclean all


RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python
