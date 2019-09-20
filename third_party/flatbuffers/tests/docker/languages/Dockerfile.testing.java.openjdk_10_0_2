FROM openjdk:10.0.2-jdk-slim-sid as base
WORKDIR /code
ADD . .
RUN cp flatc_debian_stretch flatc
WORKDIR /code/tests
RUN java -version
RUN ./JavaTest.sh
