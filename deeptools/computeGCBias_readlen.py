#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import multiprocessing
import numpy as np
import pandas as pd
import argparse
from scipy.stats import poisson
from scipy import interpolate
import py2bit
import sys
import math

from deeptoolsintervals import GTF
from deeptools.utilities import tbitToBamChrName, getGC_content
from deeptools import parserCommon, mapReduce
from deeptools.getFragmentAndReadSize import get_read_and_fragment_length
from deeptools import bamHandler

debug = 0
old_settings = np.seterr(all='ignore')


def parse_arguments(args=None):
    parentParser = parserCommon.getParentArgParse(binSize=False, blackList=True)
    requiredArgs = getRequiredArgs()
    parser = argparse.ArgumentParser(
        parents=[requiredArgs, parentParser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Computes the GC-bias using Benjamini\'s method '
        '[Benjamini & Speed (2012). Nucleic Acids Research, 40(10). doi: 10.1093/nar/gks001]. '
        'The GC-bias is visualized and the resulting table can be used to'
        'correct the bias with `correctGCBias`.',
        usage='\n computeGCBias '
        '-b file.bam --effectiveGenomeSize 2150570000 -g mm9.2bit -l 200 --GCbiasFrequenciesFile freq.txt [options]',
        conflict_handler='resolve',
        add_help=False)

    return parser


def getRequiredArgs():
    parser = argparse.ArgumentParser(add_help=False)

    required = parser.add_argument_group('Required arguments')

    required.add_argument('--bamfile', '-b',
                          metavar='bam file',
                          help='Sorted BAM file. ',
                          required=True)

    required.add_argument('--effectiveGenomeSize',
                          help='The effective genome size is the portion '
                          'of the genome that is mappable. Large fractions of '
                          'the genome are stretches of NNNN that should be '
                          'discarded. Also, if repetitive regions were not '
                          'included in the mapping of reads, the effective '
                          'genome size needs to be adjusted accordingly. '
                          'A table of values is available here: '
                          'http://deeptools.readthedocs.io/en/latest/content/feature/effectiveGenomeSize.html .',
                          default=None,
                          type=int,
                          required=True)

    required.add_argument('--genome', '-g',
                          help='Genome in two bit format. Most genomes can be '
                          'found here: http://hgdownload.cse.ucsc.edu/gbdb/ '
                          'Search for the .2bit ending. Otherwise, fasta '
                          'files can be converted to 2bit using the UCSC '
                          'programm called faToTwoBit available for different '
                          'plattforms at '
                          'http://hgdownload.cse.ucsc.edu/admin/exe/',
                          metavar='2bit FILE',
                          required=True)

    required.add_argument('--GCbiasFrequenciesFile', '-freq', '-o',
                          help='Path to save the file containing '
                          'the observed and expected read frequencies per %%GC-'
                          'content. This file is needed to run the '
                          'correctGCBias tool. This is a text file.',
                          type=argparse.FileType('w'),
                          metavar='FILE',
                          required=True)

    # define the optional arguments
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--minLength', '-min',
                          default=30,
                          help='Minimum fragment length to consider for bias computation.'
                          '(Default: %(default)s)',
                          type=int)
    optional.add_argument('--maxLength', '-max',
                          default=250,
                          help='Maximum fragment length to consider for bias computation.'
                          '(Default: %(default)s)',
                          type=int)
    optional.add_argument('--lengthStep', '-fstep',
                          default=5,
                          help='Step size for fragment lenghts between minimum and maximum fragment length.'
                          'Will be ignored if interpolate is set.'
                          '(Default: %(default)s)',
                          type=int)
    optional.add_argument("--interpolate", "-I",
                          help='Interpolates GC values and correction for missing read lengths.'
                          'This might substantially reduce computation time, but might lead to'
                          'less accurate results. Deactivated by default.',
                          action='store_true')
    optional.add_argument("--help", "-h", action="help",
                          help="show this help message and exit")

    optional.add_argument('--sampleSize',
                          default=5e7,
                          help='Number of sampling points to be considered. (Default: %(default)s)',
                          type=int)

    optional.add_argument('--extraSampling',
                          help='BED file containing genomic regions for which '
                          'extra sampling is required because they are '
                          'underrepresented in the genome.',
                          type=argparse.FileType('r'),
                          metavar='BED file')

#    plot = parser.add_argument_group('Diagnostic plot options')
#
#    plot.add_argument('--biasPlot',
#                      metavar='FILE NAME',
#                      help='If given, a diagnostic image summarizing '
#                      'the GC-bias will be saved.')
#
#    plot.add_argument('--plotFileFormat',
#                      metavar='',
#                      help='image format type. If given, this '
#                      'option overrides the '
#                      'image format based on the plotFile ending. '
#                      'The available options are: "png", '
#                      '"eps", "pdf", "plotly" and "svg"',
#                      choices=['png', 'pdf', 'svg', 'eps', 'plotly'])
#
#    plot.add_argument('--regionSize',
#                      metavar='INT',
#                      type=int,
#                      default=300,
#                      help='To plot the reads per %%GC over a region'
#                      'the size of the region is required. By default, '
#                      'the bin size is set to 300 bases, which is close to the '
#                      'standard fragment size for Illumina machines. However, '
#                      'if the depth of sequencing is low, a larger bin size '
#                      'will be required, otherwise many bins will not '
#                      'overlap with any read (Default: %(default)s)')
#
    return parser

rng = np.random.default_rng()

def roundGCLenghtBias(gc):
    gc_frac,gc_int = math.modf(round(gc*100,2))
    gc_new = gc_int + rng.binomial(1, gc_frac)
    return int(gc_new)

def getPositionsToSample(chrom, start, end, stepSize):
    """
    check if the region submitted to the worker
    overlaps with the region to take extra effort to sample.
    If that is the case, the regions to sample array is
    increased to match each of the positions in the extra
    effort region sampled at the same stepSize along the interval.

    If a filter out tree is given, then from positions to sample
    those regions are cleaned
    """
    positions_to_sample = np.arange(start, end, stepSize)

    if global_vars['filter_out']:
        filter_out_tree = GTF(global_vars['filter_out'])
    else:
        filter_out_tree = None

    if global_vars['extra_sampling_file']:
        extra_tree = GTF(global_vars['extra_sampling_file'])
    else:
        extra_tree = None

    if extra_tree:
        orig_len = len(positions_to_sample)
        try:
            extra_match = extra_tree.findOverlaps(chrom, start, end)
        except KeyError:
            extra_match = []

        if len(extra_match) > 0:
            for intval in extra_match:
                positions_to_sample = np.append(positions_to_sample,
                                                list(range(intval[0], intval[1], stepSize)))
        # remove duplicates
        positions_to_sample = np.unique(np.sort(positions_to_sample))
        if debug:
            print("sampling increased to {} from {}".format(
                len(positions_to_sample),
                orig_len))

    # skip regions that are filtered out
    if filter_out_tree:
        try:
            out_match = filter_out_tree.findOverlaps(chrom, start, end)
        except KeyError:
            out_match = []

        if len(out_match) > 0:
            for intval in out_match:
                positions_to_sample = \
                    positions_to_sample[(positions_to_sample < intval[0]) | (positions_to_sample >= intval[1])]
    return positions_to_sample


def tabulateGCcontent_wrapper(args):
#    print("ARGS:")
#    print(args)
    return tabulateGCcontent_worker(*args)


def tabulateGCcontent_worker(chromNameBam, start, end, stepSize,
                             fragmentLength,
                             chrNameBamToBit, verbose=False):
    r""" given genome regions, the GC content of the genome is tabulated for
    fragments of length 'fragmentLength' each 'stepSize' positions.

    >>> test = Tester()
    >>> args = test.testTabulateGCcontentWorker()
    >>> N_gc, F_gc = tabulateGCcontent_worker(*args)

    The forward read positions are:
    [1,  4,  10, 10, 16, 18]
    which correspond to a GC of
    [1,  1,  1,  1,  2,  1]

    The evaluated position are
    [0,  2,  4,  6,  8, 10, 12, 14, 16, 18]
    the corresponding GC is
    [2,  1,  1,  2,  2,  1,  2,  3,  2,  1]

    >>> print(N_gc)
    [0 4 5 1]
    >>> print(F_gc)
    [0 4 1 0]
    >>> test.set_filter_out_file()
    >>> chrNameBam2bit =  {'2L': 'chr2L'}

    Test for the filter out option
    >>> N_gc, F_gc = tabulateGCcontent_worker('2L', 0, 20, 2,
    ... {'median': 3}, chrNameBam2bit)
    >>> test.unset_filter_out_file()

    The evaluated positions are
    [ 0  2  8 10 12 14 16 18]
    >>> print(N_gc)
    [0 3 4 1]
    >>> print(F_gc)
    [0 3 1 0]

    Test for extra_sampling option
    >>> test.set_extra_sampling_file()
    >>> chrNameBam2bit =  {'2L': 'chr2L'}
    >>> res = tabulateGCcontent_worker('2L', 0, 20, 2,
    ... {'median': 3}, chrNameBam2bit)

    The new positions evaluated are
    [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18]
    and the GC is
    [2, 1, 1, 0, 1, 2, 2, 1,  2,  3,  2,  1]
    >>> print(res[0])
    [1 5 5 1]
    >>> print(res[1])
    [0 5 1 0]

    """
    print(f"fragmentLength: {fragmentLength}")
    if start > end:
        raise NameError("start %d bigger that end %d" % (start, end))

    chromNameBit = chrNameBamToBit[chromNameBam]

    # array to keep track of the GC from regions of length 'fragmentLength'
    # from the genome. The index of the array is used to
    # indicate the gc content. The values inside the
    # array are counts. Thus, if N_gc[10] = 3, that means
    # that 3 regions have a gc_content of 10.
    subN_gc = np.zeros(100 + 1, dtype='int') # change to percent/fraction -> len
    subF_gc = np.zeros(100 + 1, dtype='int') # change to percent/fraction -> len

    tbit = py2bit.open(global_vars['2bit'])
    bam = bamHandler.openBam(global_vars['bam'])
    peak = 0
    startTime = time.time()

    if verbose:
        print("[{:.3f}] computing positions to "
              "sample".format(time.time() - startTime))

    positions_to_sample = getPositionsToSample(chromNameBit,
                                               start, end-fragmentLength, stepSize) # substract fragment length to not exeed chrom

    read_counts = []
    # Optimize IO.
    # if the sample regions are far apart from each
    # other is faster to go to each location and fetch
    # the reads found there.
    # Otherwise, if the regions to sample are close to
    # each other, is faster to load all the reads in
    # a large region into memory and consider only
    # those falling into the positions to sample.
    # The following code gets the reads
    # that are at sampling positions that lie close together
    if np.mean(np.diff(positions_to_sample)) < 1000:
        start_pos = min(positions_to_sample)
        end_pos = max(positions_to_sample)
        if verbose:
            print("[{:.3f}] caching reads".format(time.time() - startTime))

        counts = np.bincount([r.pos - start_pos
                              for r in bam.fetch(chromNameBam, start_pos,
                                                 end_pos + 1)
                              if not r.is_reverse and not r.is_unmapped and r.pos >= start_pos],
                             minlength=end_pos - start_pos + 2)

        read_counts = counts[positions_to_sample - min(positions_to_sample)]
        if verbose:
            print("[{:.3f}] finish caching reads.".format(
                time.time() - startTime))

    countTime = time.time()

    c = 1
    for index in range(len(positions_to_sample)):
        i = positions_to_sample[index]
        # stop if the end of the chromosome is reached
        #if i + fragmentLength['median'] > tbit.chroms(chromNameBit):
        if i + fragmentLength > tbit.chroms(chromNameBit):
            c_name = tbit.chroms(chromNameBit)
            ifrag = i+fragmentLength
            print(c_name)
            print(ifrag)
            sys.stderr(f"Breaking because chrom length exeeded: {ifrag} > {c_name}")
            break

        try:
            #gc = getGC_content(tbit, chromNameBit, int(i), int(i + fragmentLength['median']), fraction=True)
            gc = getGC_content(tbit, chromNameBit, int(i), int(i + fragmentLength), fraction=True)
            #print(f"pre: {gc}")
            gc = roundGCLenghtBias(gc)
            #print(f"post: {gc}")
        except Exception as detail:
            if verbose:
                print(detail)
            continue
        #print(gc)


        # count all reads at position 'i'
        if len(read_counts) == 0:  # case when no cache was done
            num_reads = len([x.pos for x in bam.fetch(chromNameBam, i, i + 1)
                             if x.is_reverse is False and x.pos == i])
        else:
            num_reads = read_counts[index]

        if num_reads >= global_vars['max_reads'][fragmentLength]:
            peak += 1
            continue

        subN_gc[gc] += 1
        subF_gc[gc] += num_reads
        if verbose:
            if index % 50000 == 0:
                endTime = time.time()
                print("%s processing %d (%.1f per sec) @ %s:%s-%s %s" %
                      (multiprocessing.current_process().name,
                       index, index / (endTime - countTime),
                       chromNameBit, start, end, stepSize))
        c += 1

    if verbose:
        endTime = time.time()
        print("%s processing %d (%.1f per sec) @ %s:%s-%s %s" %
              (multiprocessing.current_process().name,
               index, index / (endTime - countTime),
               chromNameBit, start, end, stepSize))
        print("%s total time %.1f @ %s:%s-%s %s" % (multiprocessing.current_process().name,
                                                    (endTime - startTime), chromNameBit, start, end, stepSize))
    return(subN_gc, subF_gc)


def tabulateGCcontent(fragmentLengths, chrNameBitToBam, stepSize,
                      chromSizes, numberOfProcessors=None, verbose=False,
                      region=None):
    r"""
    Subdivides the genome or the reads into chunks to be analyzed in parallel
    using several processors. This codes handles the creation of
    workers that tabulate the GC content for small regions and then
    collects and integrates the results
    >>> test = Tester()
    >>> arg = test.testTabulateGCcontent()
    >>> res = tabulateGCcontent(*arg)
    >>> res
    array([[  0.        ,  18.        ,   1.        ],
           [  3.        ,  63.        ,   0.45815996],
           [  7.        , 159.        ,   0.42358185],
           [ 25.        , 192.        ,   1.25278115],
           [ 28.        , 215.        ,   1.25301422],
           [ 16.        , 214.        ,   0.71935396],
           [ 12.        ,  95.        ,   1.21532959],
           [  9.        ,  24.        ,   3.60800971],
           [  3.        ,  11.        ,   2.62400706],
           [  0.        ,   0.        ,   1.        ],
           [  0.        ,   0.        ,   1.        ]])
    """
    global global_vars

    chrNameBamToBit = dict([(v, k) for k, v in chrNameBitToBam.items()])
    chunkSize = int(min(2e6, 4e5 / global_vars['reads_per_bp']))
    chromSizes = [(k, v) for k, v in chromSizes if k in list(chrNameBamToBit.keys())]
    Ndict = dict()
    Fdict = dict()

    for fragmentLength in fragmentLengths:
        imap_res = mapReduce.mapReduce((stepSize,
                                        fragmentLength, chrNameBamToBit,
                                        verbose),
                                       tabulateGCcontent_wrapper,
                                       chromSizes,
                                       genomeChunkLength=chunkSize,
                                       numberOfProcessors=numberOfProcessors,
                                       region=region,
                                        verbose=verbose)
    
        for subN_gc, subF_gc in imap_res:
            try:
                F_gc += subF_gc
                N_gc += subN_gc
            except NameError:
                F_gc = subF_gc
                N_gc = subN_gc

        if sum(F_gc) == 0:
            sys.stderr.write(f"No fragments with fragment length: {fragmentLength}")
            sys.exit("No fragments included in the sampling! Consider decreasing (or maybe increasing) the --sampleSize parameter")
        #print(str(fragmentLength))
        #print(N_gc)   
        Ndict[str(fragmentLength)] = N_gc
        Fdict[str(fragmentLength)] = F_gc
        del N_gc
        del F_gc
        #print(Ndict)
    
    # create multi-index dict
    dataDict = {"N_gc_hyp_reads":Ndict,"F_gc_reads":Fdict}
    multiindex_dict = {(i,j): dataDict[i][j] 
       for i in dataDict.keys() 
       for j in dataDict[i].keys()}
    data = pd.DataFrame.from_dict(multiindex_dict, orient="index")
    data.index = pd.MultiIndex.from_tuples(data.index)
    
    #scaling = float(sum(N_gc)) / float(sum(F_gc))

    #R_gc = np.array([float(F_gc[x]) / N_gc[x] * scaling
    #                 if N_gc[x] and F_gc[x] > 0 else 1
    #                 for x in range(len(F_gc))])

    #data = np.transpose(np.vstack((F_gc, N_gc, R_gc)))
    #return imap_res
    #return data
    return data

def interpolate_ratio(df):
    # separate hypothetical read density from measured read density
    N_GC = df.loc["N_gc_hyp_reads"]
    F_GC = df.loc["F_gc_reads"]
    
    # get min and max values
    N_GC_min, N_GC_max =  np.nanmin(N_GC.index.astype("int")), np.nanmax(N_GC.index.astype("int"))
    F_GC_min, F_GC_max =  np.nanmin(F_GC.index.astype("int")), np.nanmax(F_GC.index.astype("int"))
    # sparse grid for hypothetical read density
    N_GC_readlen = N_GC.index.to_numpy(dtype=int)
    N_GC_gc = N_GC.columns.to_numpy(dtype=int)
    N_xx,N_yy = np.meshgrid(N_GC_gc,N_GC_readlen)
    
    # sparse grid for measured read density
    F_GC_readlen = F_GC.index.to_numpy(dtype=int)
    F_GC_gc = F_GC.columns.to_numpy(dtype=int)
    F_xx,F_yy = np.meshgrid(F_GC_gc,F_GC_readlen)
    
    # determine nans for sparse data
    N_nans = np.isnan(N_GC.to_numpy())
    F_nans = np.isnan(F_GC.to_numpy())
    
    # determine nans for sparse data
    N_nans = np.isnan(N_GC.to_numpy())
    F_nans = np.isnan(F_GC.to_numpy())
    
    # Select non_NaN coordinates
    N_X = N_xx[~N_nans]
    N_Y = N_yy[~N_nans]
    F_X = F_xx[~F_nans]
    F_Y = F_yy[~F_nans]
    #reshape sparse coordinates to shape (N, 2) in 2d
    N_sparse_points = np.stack([N_X,N_Y],-1)
    F_sparse_points = np.stack([F_X,F_Y],-1)
    
    N_zz = N_GC.to_numpy()[~N_nans]
    F_zz = F_GC.to_numpy()[~F_nans]
    
    N_f2 = interpolate.RBFInterpolator(N_sparse_points,N_zz,kernel="quintic", smoothing=0.1)
    F_f2 = interpolate.RBFInterpolator(F_sparse_points,F_zz,kernel="quintic", smoothing=0.1)
    
    scaling_dict = dict()
    for i in np.arange(N_GC_min,N_GC_max+1,1):
        ref_a,ref_b = np.meshgrid(N_GC.columns.to_numpy(dtype=int),i)
        ref_dense = np.stack([ref_a.ravel(), ref_b.ravel()], -1)
        N_tmp = N_f2(ref_dense).reshape(ref_a.shape)
        F_tmp = F_f2(ref_dense).reshape(ref_a.shape)
        scaling_dict[i] = float(np.sum(N_tmp)) / float(np.sum(F_tmp))
    
    # get dense data (full GC and readlen range)
    N_a,N_b = np.meshgrid(N_GC.columns.to_numpy(dtype=int), np.arange(N_GC_min,N_GC_max+1,1))
    F_a,F_b = np.meshgrid(F_GC.columns.to_numpy(dtype=int), np.arange(F_GC_min,F_GC_max+1,1))
    # convert to 2D coordinate pairs
    N_dense_points = np.stack([N_a.ravel(), N_b.ravel()], -1)
    F_dense_points = np.stack([F_a.ravel(), F_b.ravel()], -1)
    
    r_list = list()
    for i in N_dense_points:
        x = i.reshape(1,2)
        scaling = scaling_dict[x[0][1]]
        if N_f2(x) > 0 and F_f2(x) > 0:
            ratio = float(F_f2(x) / N_f2(x) * scaling)
        else:
            ratio = 1
        r_list.append(ratio)
    
    ratio_dense = np.array(r_list).reshape(N_a.shape)
    ind = pd.MultiIndex.from_product([["R_gc"], np.arange(N_GC_min,N_GC_max+1,1)])
    
    return pd.DataFrame(ratio_dense,columns=N_GC.columns, index=ind)

def get_ratio(df):
    # separate hypothetical read density from measured read density
    N_GC = df.loc["N_gc_hyp_reads"]
    F_GC = df.loc["F_gc_reads"]
    # get min and max values
    N_GC_min, N_GC_max =  np.nanmin(N_GC.index), np.nanmax(N_GC.index)
    F_GC_min, F_GC_max =  np.nanmin(F_GC.index), np.nanmax(F_GC.index)
    
    scaling_dict = dict()
    for i in np.arange(N_GC_min,N_GC_max+1,1):
        N_tmp = N_GC.loc[i].to_numpy()
        F_tmp = F_GC.loc[i].to_numpy()
        scaling_dict[i] = float(np.sum(N_tmp)) / float(np.sum(F_tmp))

    r_dict = dict()
    for i in np.arange(N_GC_min,N_GC_max+1,1):
        scaling = scaling_dict[i]
        F_gc_t = F_GC.loc[i]
        N_gc_t = N_GC.loc[i]
        R_gc_t = np.array([float(F_gc_t[x]) / N_gc_t[x] * scaling
                         if N_gc_t[x] and F_gc_t[x] > 0 else 1
                         for x in range(len(F_gc_t))])
        r_dict[i] = R_gc_t
    
    ratio_dense = pd.DataFrame.from_dict(r_dict, orient="index", columns=N_GC.columns)
    ind = pd.MultiIndex.from_product([["R_gc"], ratio_dense.index])
    ratio_dense.index = ind
    
    return ratio_dense

def main(args=None):
    args = parse_arguments().parse_args(args)

    if args.extraSampling:
        extra_sampling_file = args.extraSampling.name
        args.extraSampling.close()
    else:
        extra_sampling_file = None

    global global_vars
    global_vars = {}
    global_vars['2bit'] = args.genome
    global_vars['bam'] = args.bamfile
    global_vars['filter_out'] = args.blackListFileName
    global_vars['extra_sampling_file'] = extra_sampling_file

    tbit = py2bit.open(global_vars['2bit'])
    bam, mapped, unmapped, stats = bamHandler.openBam(global_vars['bam'], returnStats=True, nThreads=args.numberOfProcessors)

    if args.interpolate:
        length_step = args.lengthStep
    else:
        length_step = 1

    fragmentLengths = np.arange(args.minLength,args.maxLength+1,length_step).tolist()

    chrNameBitToBam = tbitToBamChrName(list(tbit.chroms().keys()), bam.references)

    global_vars['genome_size'] = sum(tbit.chroms().values())
    global_vars['total_reads'] = mapped
    global_vars['reads_per_bp'] = \
        float(global_vars['total_reads']) / args.effectiveGenomeSize

    confidence_p_value = float(1) / args.sampleSize

    # chromSizes: list of tuples
    chromSizes = [(bam.references[i], bam.lengths[i])
                  for i in range(len(bam.references))]
    #chromSizes = [x for x in chromSizes if x[0] in tbit.chroms()] # why would you do this? There is a mapping specifically instead of tbut.chroms()

    max_read_dict = dict()
    min_read_dict = dict()
    for fragment_len in fragmentLengths:
        # use poisson distribution to identify peaks that should be discarted.
        # I multiply by 4, because the real distribution of reads
        # vary depending on the gc content
        # and the global number of reads per bp may a be too low.
        # empirically, a value of at least 4 times as big as the
        # reads_per_bp was found.
        # Similarly for the min value, I divide by 4.
        max_read_dict[fragment_len] = poisson(4 * global_vars['reads_per_bp'] * fragment_len).isf(confidence_p_value)
        # this may be of not use, unless the depth of sequencing is really high
        # as this value is close to 0
        min_read_dict[fragment_len] = poisson(0.25 * global_vars['reads_per_bp'] * fragment_len).ppf(confidence_p_value)

    global_vars['max_reads'] = max_read_dict
    global_vars['min_reads'] = min_read_dict
    
    for key in global_vars:
        print("{}: {}".format(key, global_vars[key]))

    print("computing frequencies")
    # the GC of the genome is sampled each stepSize bp.
    stepSize = max(int(global_vars['genome_size'] / args.sampleSize), 1)
    print("stepSize for genome sampling: {}".format(stepSize))

    data = tabulateGCcontent(fragmentLengths,
                             chrNameBitToBam, stepSize,
                             chromSizes,
                             numberOfProcessors=args.numberOfProcessors,
                             verbose=args.verbose,
                             region=args.region)

    if args.interpolate:
        r_data = interpolate_ratio(data)
    else:
        r_data = get_ratio(data)
    out_data = data.append(r_data)
    out_data.to_csv(args.GCbiasFrequenciesFile.name, sep="\t")



if __name__ == "__main__":
    main()
