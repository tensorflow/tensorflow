FROM golang:1.11-stretch as base
WORKDIR /code
ADD . .
RUN cp flatc_debian_stretch flatc
WORKDIR /code/tests
RUN go version
RUN ./GoTest.sh
