FROM php:7.3-cli-stretch as base
WORKDIR /code
ADD . .
RUN cp flatc_debian_stretch flatc
WORKDIR /code/tests
RUN php --version
RUN php phpTest.php
RUN sh phpUnionVectorTest.sh
