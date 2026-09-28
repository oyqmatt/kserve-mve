#vars
IMAGENAME=kserve_img
REPO=localhost
IMAGEFULLNAME=${REPO}/${IMAGENAME}

.PHONY: help build

help:
	    @echo "Makefile arguments:"
	    @echo ""
	    @echo "build"

.DEFAULT_GOAL := build

build:
	    @docker build -f docker/Dockerfile -t ${IMAGEFULLNAME}:latest .
