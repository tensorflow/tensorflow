# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
ARG TF_PACKAGE=tensorflow
RUN ${PIP} install ${TF_PACKAGE}
