RUN ${PIP} install jupyter matplotlib
RUN ${PIP} install jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
RUN apt-get install -y --no-install-recommends wget
WORKDIR /tf/tensorflow-tutorials
RUN wget https://raw.githubusercontent.com/tensorflow/docs/r1.15/site/en/tutorials/keras/basic_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/r1.15/site/en/tutorials/keras/basic_text_classification.ipynb
COPY readme-for-jupyter.md README.md
RUN apt-get autoremove -y && apt-get remove -y wget
WORKDIR /tf
EXPOSE 8888

RUN ${PYTHON} -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
