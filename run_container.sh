#!/bin/bash
# Run container
# Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

INPATH=${INPATH:-"/arch/radar/raw/"}
OUTPATH=${OUTPATH:-"output"}
CODE=${CODE:-"$(pwd)"}

echo INPATH: "$INPATH"
echo OUTPATH: "$OUTPATH"
echo CODE: "$CODE"
echo "Conda environment: wtc"

# Run with volume mounts
docker run -it \
    --entrypoint /bin/bash \
    --rm \
    -v "${INPATH}":/input \
    -v "${OUTPATH}":/output \
    -v "${CODE}":/code \
    -w /code \
    -e PYART_CONFIG=/code/.pyart_config.py \
    wtcfiltering:latest
