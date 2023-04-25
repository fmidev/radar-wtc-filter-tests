FROM ubuntu:20.04

# Install conda
RUN apt-get -qq update && apt-get -qq -y install curl bzip2\
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Create the environment:
ARG conda_env=wtc
COPY environment.yml .
ENV PYTHONDONTWRITEBYTECODE=true
RUN conda install -c conda-forge mamba && \
    mamba env create -f environment.yml -n $conda_env && \
    mamba clean --all -f -y

RUN conda init bash

# Allow environment to be activated
RUN echo "conda activate wtc" >> ~/.profile
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env

COPY . /code
WORKDIR /code

ENV INPUT_PATH /input
ENV FILTER_COND filters.yaml

ENTRYPOINT conda run -n $conda_env python plot_ppis.py
