#!/bin/bash

git config --global user.email "g.ortega.david@gmail.com"
git config --global user.name "DavidGOrtega"

EXP_NAME=cml-run-${GITHUB_SHA}-2
EXP_AVAIL=$(dvc exp pull --run-cache origin $EXP_NAME &>/dev/null)
if [[ -z "$EXP_AVAIL" ]]; then
    echo "############\nFirst Time\n############"
    dvc exp run -n $EXP_NAME --pull
else    
    echo "############\nResuming\n############"
    dvc exp apply $EXP_NAME
    dvc exp run
fi