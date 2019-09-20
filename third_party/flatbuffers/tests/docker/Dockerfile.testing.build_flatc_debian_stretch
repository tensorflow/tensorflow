FROM debian:9.6-slim as base
RUN apt -qq update >/dev/null
RUN apt -qq install -y cmake make build-essential >/dev/null
FROM base
WORKDIR /code
ADD . .
RUN cmake -G "Unix Makefiles"
RUN make flatc
RUN ls flatc
