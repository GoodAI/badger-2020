FROM python:3.7
#EXPOSE 5000

WORKDIR /usr/src/app

COPY requirements_base.txt ./
RUN pip install --no-cache-dir -r requirements_base.txt

COPY requirements.txt ./

# note: the --trusted-host address is necessary for pulling this repository from the server
RUN pip install --no-cache-dir -r requirements.txt --trusted-host 192.168.120.2

# better to copy just selected folders/files
#COPY . .

# support for NVIDIA drivers in the docker
# installation according to this:
#   https://github.com/NVIDIA/nvidia-docker
# paths from here:
#   https://medium.com/@adityathiruvengadam/cuda-docker-%EF%B8%8F-for-deep-learning-cab7c2be67f9
# usage after standard build:
#   docker run --gpus all repo:img bash run_aws.sh
ENV PATH /usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

# https://serverfault.com/questions/869055/how-to-set-environment-variable-within-the-docker-file
ENV PYTHONPATH="$PYTHONPATH:."

# optional, shell script runs the 'run_aws.sh' instead
#CMD [ "python", "./example_train.py" ]

