FROM node:11.2.0-stretch as base
WORKDIR /code
ADD . .
RUN cp flatc_debian_stretch flatc
WORKDIR /code/tests
RUN node --version
RUN ../flatc -b -I include_test monster_test.fbs unicode_test.json
RUN node JavaScriptTest ./monster_test_generated
