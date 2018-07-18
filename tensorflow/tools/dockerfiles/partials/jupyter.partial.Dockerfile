RUN ${PIP} install jupyter

RUN mkdir /notebooks && chmod 777 /notebooks
RUN mkdir /.local && chmod 777 /.local
WORKDIR /notebooks
EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root"]
