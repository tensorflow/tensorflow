FROM python:3.7.1-slim-stretch as base
WORKDIR /code
ADD . .
RUN cp flatc_debian_stretch flatc
WORKDIR /code/tests
RUN python --version
RUN pip install coverage
RUN ./PythonTest.sh
