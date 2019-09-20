FROM rust:1.30.1-slim-stretch as base
WORKDIR /code
ADD . .
RUN cp flatc_debian_stretch flatc
WORKDIR /code/tests
RUN rustc --version
RUN ./RustTest.sh
