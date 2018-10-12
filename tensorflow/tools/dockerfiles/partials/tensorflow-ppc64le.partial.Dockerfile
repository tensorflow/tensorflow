ARG TF_PACKAGE=tensorflow-gpu
RUN apt-get update && apt-get install -y wget
RUN if [ ${TF_PACKAGE} = tensorflow-gpu ]; then \
        BASE=https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/lastSuccessfulBuild/; \
    elif [ ${TF_PACKAGE} = tf-nightly-gpu ]; then \
        BASE=https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Nightly_Artifact/lastSuccessfulBuild/; \
    fi; \
    MINOR=`${PYTHON} -c 'import sys; print(sys.version_info[1])'`; \
    PACKAGE=$(wget -qO- ${BASE}"api/xml?xpath=//fileName&wrapper=artifacts" | grep -o "[^<>]*cp${_PY_SUFFIX}${MINOR}[^<>]*.whl"); \
    wget ${BASE}"artifact/tensorflow_pkg/"${PACKAGE}; \
    ${PIP} install ${PACKAGE}
