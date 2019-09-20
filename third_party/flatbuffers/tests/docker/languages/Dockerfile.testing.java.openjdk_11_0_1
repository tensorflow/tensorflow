FROM openjdk:11.0.1-jdk-slim-sid as base
WORKDIR /code
ADD . .
RUN cp flatc_debian_stretch flatc
WORKDIR /code/tests
RUN java -version
RUN ./JavaTest.sh
