#!/bin/bash

REPO=snail.goodai.com:5000
NAME=badger-utils
VERSION=0.0.13

NAME_LATEST="${REPO}/${NAME}:latest"
NAME_VERSION="${REPO}/${NAME}:${VERSION}"

#VERSION=$(git rev-parse --short HEAD)

print_and_run() {
  echo $*
  $*
}



#CMD="docker image build -t ${NAME} . "

print_and_run docker image build -t ${NAME} . --no-cache

print_and_run docker tag ${NAME} ${NAME_LATEST}
#print_and_run docker tag ${NAME} ${NAME_VERSION}

print_and_run docker push ${NAME_LATEST}


#CMD="docker image build -t ${REPO}/${NAME}:${VERSION} . --no-cache"
#CMD="docker image build -t ${REPO}/${NAME}:${VERSION} . --no-cache"
#echo $CMD
#$CMD

#CMD="docker push ${REPO}/${NAME}:${VERSION}"
#echo $CMD
#$CMD


# Uncomment when pushing new version
#docker image build -t ${REPO}/${NAME}:${VERSION} .
#docker push ${REPO}/${NAME}:${VERSION}
