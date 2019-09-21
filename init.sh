#!/usr/bin/env bash

# create venv
python3 -m venv env

# activate
source env/bin/activate

# install required packages
pip install -r requirements.txt