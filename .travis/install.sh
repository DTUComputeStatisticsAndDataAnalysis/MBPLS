#!/bin/bash

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda.sh;
  bash ~/miniconda.sh -b -p $HOME/miniconda;
  export PATH="$HOME/miniconda/bin:$PATH";
  conda config --set always_yes yes --set changeps1 no;
  conda update -q conda;
  conda info -a;
  case "${CONDAENV}" in
    3.5)
      conda create -q -n testenvironment python=3.5
      ;;
    3.6)
      conda create -q -n testenvironment python=3.6
      ;;
    3.7)
      conda create -q -n testenvironment python=3.7
      ;;
  esac
  source activate testenvironment;
  conda install pytest;
  conda install pip;
fi