#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from deeptools._version import __version__


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
cfDNA_GCcorrection is a suite of python tools particularly developed for the efficient 
analysis and correction of GC bias in cfDNA sequencing data. 

[ Tools for QC ]
    computeGCBias_background        computes the GC bias of a reference genome
    computeGCBias_readlen           computes the length based GC bias of a sample


[ Tools for BAM and bigWig file processing ]
    correctGCBias_readlen           corrects GC bias from bam file by attaching weight tags 
                                    or changing read copies. 


""")

    parser.add_argument('--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    return parser


def process_args(args=None):
    args = parse_arguments().parse_args(args)

    return args


def main(args=None):
    if args is None and len(sys.argv) == 1:
        args = ["--help"]
    process_args(args)
