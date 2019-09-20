FROM mono:5.18 as base
WORKDIR /code
ADD . .
RUN cp flatc_debian_stretch flatc
WORKDIR /code/tests
RUN mono --version
WORKDIR /code/tests/FlatBuffers.Test
RUN sh NetTest.sh
