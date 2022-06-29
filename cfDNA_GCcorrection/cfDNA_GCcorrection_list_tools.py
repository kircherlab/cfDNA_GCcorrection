#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from deeptools._version import __version__


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
deepTools is a suite of python tools particularly developed for the efficient analysis of
high-throughput sequencing data, such as ChIP-seq, RNA-seq or MNase-seq.

Each tool should be called by its own name as in the following example:

 $ bamCoverage -b reads.bam -o coverage.bw

[ Tools for BAM and bigWig file processing ]
    correctGCBias           corrects GC bias from bam file. Don't use it with ChIP data

[ Tools for QC ]
    computeGCBias           computes and plots the GC bias of a sample


For more information visit: http://deeptools.readthedocs.org
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
