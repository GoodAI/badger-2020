# Base docker image for badger_utils

Base docker image containing Python and badger_utils.

All common libraries like torch, sacred, matplotlib, bokeh, ... are included

## Installation

Docker repository `snail.goodai.com:5000` uses self-signed SSL certificate and 
it is needed to add it to Docker trusted CAs by running `add_cert.sh`


## Update image

`deploy.sh` - just run this file - it will build the image and push it 
to the `snail.goodai.com:5000` docker repository 
