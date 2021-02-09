# On CentOS 7, yum needs to run with Python2.7
RUN sed -i 's#/usr/bin/python#/usr/bin/python2#g' /usr/bin/yum /usr/libexec/urlgrabber-ext-down
