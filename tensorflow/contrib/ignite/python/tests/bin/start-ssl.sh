#!/usr/bin/env bash

nohup apache-ignite-fabric/bin/ignite.sh /data/config/ignite-config-ssl.xml & 
sleep 5 # Wait Apache Ignite to be started
./apache-ignite-fabric/bin/sqlline.sh -u 'jdbc:ignite:thin://127.0.0.1/?sslMode=require&sslClientCertificateKeyStoreUrl=/data/keystore/client.jks&sslClientCertificateKeyStorePassword=123456&sslTrustAll=true' --run=/data/sql/init.sql --verbose=true
#tail -f nohup.out
bash
