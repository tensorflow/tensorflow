FROM rust:1.30.1-slim-stretch as base
RUN apt -qq update -y && apt -qq install -y \
    gcc-mips-linux-gnu \
    libexpat1 \
    libmagic1 \
    libmpdec2 \
    libreadline7 \
    qemu-user
RUN rustup target add mips-unknown-linux-gnu
WORKDIR /code
ADD . .
RUN cp flatc_debian_stretch flatc
WORKDIR /code/tests
RUN rustc --version
RUN ./RustTest.sh mips-unknown-linux-gnu
