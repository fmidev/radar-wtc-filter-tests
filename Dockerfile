FROM mambaorg/micromamba:1-alpine

# Change user, from https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html#changing-the-user-id-or-name
ARG NEW_MAMBA_USER=mambauser
ARG NEW_MAMBA_USER_ID=1000
ARG NEW_MAMBA_USER_GID=1000
USER root

RUN if grep -q '^ID=alpine$' /etc/os-release; then \
    # alpine does not have usermod/groupmod
    apk add --no-cache --virtual temp-packages shadow=4.13-r0; \
    fi && \
    usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" \
    --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}" && \
    groupmod "--new-name=${NEW_MAMBA_USER}" \
    "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}" && \
    if grep -q '^ID=alpine$' /etc/os-release; then \
    # remove the packages that were only needed for usermod/groupmod
    apk del temp-packages; \
    fi && \
    # Update the expected value of MAMBA_USER for the
    # _entrypoint.sh consistency check.
    echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" && \
    :
ENV MAMBA_USER=$NEW_MAMBA_USER
USER $MAMBA_USER

# Create conda environment
ENV conda_env base
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/env.yaml
ENV PYTHONDONTWRITEBYTECODE=true
# For reference for cleanup, see https://uwekorn.com/2021/03/01/deploying-conda-environments-in-docker-how-to-do-it-right.html
# and https://jcristharif.com/conda-docker-tips.html
RUN micromamba install -y --file env.yaml --freeze-installed \
    && micromamba clean -afy \
    && find -name '*.a' -delete \
    && rm -rf /opt/conda/conda-meta \
    && rm -rf /opt/conda/include \
    && rm /opt/conda//lib/libpython*.so.1.0 \
    && find -name '__pycache__' -type d -exec rm -rf '{}' '+' \
    && rm -rf /opt/conda/lib/python3.9/site-packages/pip /opt/conda/lib/python3.9/idlelib /opt/conda/lib/python3.9/ensurepip \
    /opt/conda/lib/libasan.so.5.0.0 \
    /opt/conda/lib/libtsan.so.0.0.0 \
    /opt/conda/lib/liblsan.so.0.0.0 \
    /opt/conda/lib/libubsan.so.1.0.0 \
    /opt/conda/bin/x86_64-conda-linux-gnu-ld \
    /opt/conda/bin/sqlite3 \
    /opt/conda/bin/openssl \
    /opt/conda/share/terminfo \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete \
    && find /opt/conda/lib/python*/site-packages/scipy -name 'tests' -type d -exec rm -rf '{}' '+' \
    && find /opt/conda/lib/python*/site-packages/numpy -name 'tests' -type d -exec rm -rf '{}' '+' \
    && find /opt/conda/lib/python*/site-packages/pandas -name 'tests' -type d -exec rm -rf '{}' '+' \
    && find /opt/conda/lib/python*/site-packages -name '*.pyx' -delete \
    && rm -rf /opt/conda/lib/python*/site-packages/uvloop/loop.c

# Allow environment to be activated
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env

COPY --chown=$MAMBA_USER:$MAMBA_USER ./src /src
WORKDIR /src

ENTRYPOINT /bin/bash
