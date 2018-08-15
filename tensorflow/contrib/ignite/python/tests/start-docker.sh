#!/usr/bin/env bash

IGNITE_VERSION=2.6.0

docker run -itd --name ignite-plain -p 42300:10800 -v `pwd`:/data apacheignite/ignite:${IGNITE_VERSION} \
  /data/bin/start-plain.sh
docker run -itd --name ignite-ssl -p 42301:10800 -v `pwd`:/data apacheignite/ignite:${IGNITE_VERSION} \
  /data/bin/start-ssl.sh
docker run -itd --name ignite-ssl-auth -p 42302:10800 -v `pwd`:/data apacheignite/ignite:${IGNITE_VERSION} \
  /data/bin/start-ssl-auth.sh

