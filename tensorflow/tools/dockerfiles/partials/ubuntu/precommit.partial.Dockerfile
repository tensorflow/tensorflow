RUN python3 -m pip --no-cache-dir install \
        pylint==2.7.4 \
        pre-commit
RUN if [ "$CHECKOUT_TF_SRC" = 1 ] ; then cd /tensorflow_src && pre-commit install ; fi