#!/usr/bin/env python3

import sys
import argparse
from argparse import RawTextHelpFormatter

import src.main as main

desc="""PEIM - Porous Electrode Impednace Model"""

parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
args = parser.parse_args()

try:
    main.main(sys.argv[1], sys.argv[2])
except IndexError:
    print("ERROR: No parameter file specified. Aborting")
raise