#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math
import os
import random
import subprocess
import sys
import time
from itertools import starmap

import click
import numpy as np
import pandas as pd
import py2bit
import pysam
from loguru import logger
from mpire import WorkerPool
from mpire.utils import make_single_arguments
from scipy.stats import binom

from cfDNA_GCcorrection.bamHandler import openBam
from cfDNA_GCcorrection.mapReduce import getUserRegion
from cfDNA_GCcorrection.utilities import getGC_content, getTempFileName, map_chroms


def run_shell_command(command):
    """
    Runs the given shell command. Report
    any errors found.
    """
    try:
        subprocess.check_call(command, shell=True)

    except subprocess.CalledProcessError as error:
        logger.error(f"Error{error}\n")
        exit(1)
    except Exception as error:
        logger.error(f"Error: {error}\n")
        exit(1)


def roundGCLenghtBias(gc):
    value = gc * 100
    gc_new = int(value) + (1 if np.random.rand() < value % 1 else 0)
    return gc_new


# def getReadGCcontent(tbit, read, fragmentLength, chrNameBit):
def getReadGCcontent(tbit, read, chrNameBit):  # fragmentLength not needed anymore
    """
    The fragments for forward and reverse reads are defined as follows::

           |- read.pos       |- read.aend
        ---+=================>-----------------------+---------    Forward strand

           |-fragStart                               |-fragEnd

        ---+-----------------------<=================+---------    Reverse strand
                                   |-read.pos        |-read.aend

           |-----------------------------------------|
                            read.tlen

    """
    fragStart = None
    fragEnd = None

    if read.is_paired and read.is_proper_pair:  # and abs(read.tlen) < 2 * fragmentLength:
        if read.is_reverse and read.tlen < 0:
            fragEnd = read.reference_end
            fragStart = read.reference_end + read.template_length
        elif read.template_length >= read.query_alignment_length:
            fragStart = read.pos
            fragEnd = read.pos + read.template_length

    if not fragStart:
        if read.is_reverse:
            fragEnd = read.reference_end
            fragStart = read.reference_start  # read.reference_end - fragmentLength
        else:
            fragStart = read.reference_start  # read.pos
            fragEnd = read.reference_end  # fragStart + fragmentLength
    fragStart = max(0, fragStart)
    try:
        gc = getGC_content(tbit, chrNameBit, fragStart, fragEnd, fraction=True)
        gc = roundGCLenghtBias(gc)
    except Exception:
        return None
    if gc is None:
        return None
    # match the gc to the given fragmentLength
    # gc = int(np.round(gc * fragmentLength))
    return gc


def numCopiesOfRead(value):
    """
    Based int he R_gc value, decides
    whether to keep, duplicate, triplicate or delete the read.
    It returns an integer, that tells the number of copies of the read
    that should be keep.
    >>> np.random.seed(1)
    >>> numCopiesOfRead(0.8)
    1
    >>> numCopiesOfRead(2.5)
    2
    >>> numCopiesOfRead(None)
    1
    """
    copies = 1
    if value:
        copies = int(value) + (1 if np.random.rand() < value % 1 else 0)
    return copies


def findNearestIndex(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def writeCorrectedSam_wrapper(args):
    return writeCorrectedSam_worker(*args)


def writeCorrectedBam_worker(
    R_gc_dict,
    bam_file,
    twobit_file,
    chrNameBam,
    chrNameBit,
    start,
    end,
    max_dup_gc=None,
    tag_but_not_change_number=True,
    verbose=False,
    debug=False,
    threads=10,
    default_value=1,
    use_nearest_weight=False,
):
    r"""
    Writes a BAM file, deleting and adding some reads in order to compensate
    for the GC bias, if tag_but_not_change_number is False.
    **This is a stochastic method.** Otherwise, all alignments get a YC and a YG tag
     and are written to a new file containing the same amount of alns.
    """

    if debug:
        logger.debug("Sam for %s %s %s " % (chrNameBit, start, end))
    i = 0

    tbit = py2bit.open(twobit_file)

    # We want to speed up I/O operations for pysam by specifying threads to the AlignmentFile API.
    # For optimization purposes, we want more compression threads for the output file.
    # See https://github.com/pysam-developers/pysam/pull/638#issue-302695163
    out_threads = math.ceil(threads * 2 / 3)
    in_threads = max(1, (threads - out_threads))

    bam = pysam.AlignmentFile(bam_file, "rb", threads=in_threads)
    tempFileName = getTempFileName(suffix=".bam")
    outfile = pysam.AlignmentFile(tempFileName, "wb", template=bam, threads=out_threads)

    R_gc_lengths = np.asarray(list(R_gc_dict.keys()))
    startTime = time.time()
    matePairs = {}
    read_repetitions = 0
    removed_duplicated_reads = 0

    ## general processing
    reads = 0
    pread = None

    for read in bam.fetch(chrNameBam, start, end):
        if read.pos <= start or read.is_unmapped:
            continue
        reads += 1
        read_name = read.qname
        if (
            read.is_paired
        ):  # As proper pairs highly depend on the mapping software, we only filter downstream for respective length!
            r_len = abs(read.template_length)
        else:
            r_len = read.query_length
        try:
            # copies = matePairs[read_name]['copies']
            gc = matePairs[read_name]["gc"]
            if tag_but_not_change_number:
                del matePairs[read_name]
        except:
            # this exception happens when a mate is
            # not present. This could
            # happen because of removal of the mate
            # by some filtering
            gc = getReadGCcontent(tbit, read, chrNameBit)
            if debug:
                logger.debug(
                    f"writeCorrectedSam_worker;read_name: {read_name}; gc:{gc}"
                )

        if gc:
            gc_for_tag = gc  # int(100 * np.round(float(gc) / fragmentLength,
            #                   decimals=2))
            try:
                # ('YC', float(round(float(1) / R_gc_dict[gc], 2)), "f"))
                # readTag.append(
                #    ('YC', R_gc_dict[r_len][gc], "f")
                # )
                yc_tag = ("YC", R_gc_dict[r_len][gc], "f")
            except KeyError as e:
                if use_nearest_weight:
                    r_len = findNearestIndex(R_gc_lengths, r_len)
                    if debug:
                        logger.debug(
                            f"Weight: Read length {e} was not in correction table. \
                                Correction was done with closest available read length: {r_len}"
                        )
                    yc_tag = ("YC", R_gc_dict[r_len][gc], "f")
                else:
                    if debug:
                        logger.debug(
                            f"Weight: Read length {e} was not in correction table. \
                                Correction was done with the default value: {default_value}"
                        )
                    yc_tag = ("YC", default_value, "f")
            read.set_tag(*yc_tag)
        else:
            gc_for_tag = -1

        # yg_tag = ('YG', gc_for_tag, "i")
        read.set_tag("YG", gc_for_tag, "i")

        if tag_but_not_change_number:
            if read.is_paired and not read.mate_is_unmapped and not read.is_reverse:
                matePairs[read_name] = {"gc": gc}
            outfile.write(read)
            if debug:
                if i % 350000 == 0 and i > 0:
                    endTime = time.time()
                    logger.debug(
                        f"Processing {i} reads ({i / (endTime - startTime):.1f} per sec) @ {chrNameBit}:{start}-{end}"
                    )
            i += 1
            continue

        # Everything below is only executed if copies or reads are created

        try:
            copies = matePairs[read_name]["copies"]
            # gc = matePairs[read_name]['gc']
            del matePairs[read_name]
        except:
            # this exception happens when a mate is
            # not present. This could
            # happen because of removal of the mate
            # by some filtering
            if gc:
                try:
                    copies = numCopiesOfRead(R_gc_dict[r_len][gc])
                except KeyError as e:
                    if use_nearest_weight:
                        r_len = findNearestIndex(R_gc_lengths, r_len)
                        if debug:
                            logger.debug(
                                f"Copies: Read length {e} was not in correction table. \
                                    Correction was done with closest available read length: {r_len} "
                            )
                        copies = numCopiesOfRead(R_gc_dict[r_len][gc])
                    else:
                        if debug:
                            logger.debug(
                                f"Copies: Read length {e} was not in correction table. \
                                    Copies was set to 1"
                            )
                        copies = 1
            else:
                copies = 1
        # is this read in the same orientation and position as the previous?
        if (
            gc
            and reads > 1
            and read.pos == pread.pos
            and read.is_reverse == pread.is_reverse
            and read.pnext == pread.pnext
        ):
            read_repetitions += 1
            try:
                tmp_max_dup_gc = max_dup_gc[r_len][gc]
            except KeyError as e:
                if use_nearest_weight:
                    r_len = findNearestIndex(R_gc_lengths, r_len)
                    if debug:
                        logger.debug(
                            f"Max_dup_copies: Read length {e} was not in correction table. \
                                Correction was done with closest available read length: {r_len}"
                        )
                    tmp_max_dup_gc = max_dup_gc[r_len][gc]
                else:
                    if debug:
                        logger.debug(
                            f"Max_dup_copies: Read length {e} was not in correction table. \
                                Max_dup_copies were set to 1"
                        )
                    tmp_max_dup_gc = 1
            if read_repetitions >= tmp_max_dup_gc:
                copies = 0  # in other words do not take into account this read
                removed_duplicated_reads += 1
        else:
            read_repetitions = 0

        if read.is_paired and not read.mate_is_unmapped and not read.is_reverse:
            matePairs[read_name]["copies"] = copies
        pread = copy.copy(read)  # copy read for calculating read repetitions
        for numCop in range(1, copies + 1):
            # the read has to be renamed such that newly
            # formed pairs will match
            if numCop > 1:
                read.qname = read_name + "_%d" % numCop
            outfile.write(read)

        if debug:
            if i % 350000 == 0 and i > 0:
                endTime = time.time()
                logger.debug(
                    f"Processing {i} reads ({i / (endTime - startTime):.1f} per sec) @ {chrNameBit}:{start}-{end}"
                )
        i += 1

    # finish up process
    outfile.close()
    if verbose:
        endTime = time.time()
        logger.info(
            f"Processed {i} reads ({i / (endTime - startTime):.1f} per sec) @ {chrNameBit}:{start}-{end}"
        )

        if not tag_but_not_change_number:  # return only if read copies were changed
            percentage = (
                float(removed_duplicated_reads) * 100 / reads if reads > 0 else 0
            )
            logger.info(
                "duplicated reads removed %d of %d (%.2f) "
                % (removed_duplicated_reads, reads, percentage)
            )

    return tempFileName


# def getFragmentFromRead(read, defaultFragmentLength, extendPairedEnds=True):
#    """
#    The read has to be pysam object.
#
#    The following values are defined (for forward reads)::
#
#
#             |--          -- read.tlen --              --|
#             |-- read.alen --|
#        -----|===============>------------<==============|----
#             |               |            |
#          read.pos      read.aend      read.pnext
#
#
#          and for reverse reads
#
#
#             |--             -- read.tlen --           --|
#                                         |-- read.alen --|
#        -----|===============>-----------<===============|----
#             |                           |               |
#          read.pnext                   read.pos      read.aend
#
#    this is a sketch of a pair-end reads
#
#    The function returns the fragment start and end, either
#    using the paired end information (if available) or
#    extending the read in the appropriate direction if this
#    is single-end.
#
#    Parameters
#    ----------
#    read : pysam read object
#
#
#    Returns
#    -------
#    tuple
#        (fragment start, fragment end)
#
#    """
#    # convert reads to fragments
#
#    # this option indicates that the paired ends correspond
#    # to the fragment ends
#    # condition read.tlen < maxPairedFragmentLength is added to avoid read pairs
#    # that span thousands of base pairs
#
#    if extendPairedEnds is True and read.is_paired and 0 < abs(read.tlen) < 1000:
#        if read.is_reverse:
#            fragmentStart = read.pnext
#            fragmentEnd = read.aend
#        else:
#            fragmentStart = read.pos
#            # the end of the fragment is defined as
#            # the start of the forward read plus the insert length
#            fragmentEnd = read.pos + read.tlen
#    else:
#        if defaultFragmentLength <= read.aend - read.pos:
#            fragmentStart = read.pos
#            fragmentEnd = read.aend
#        else:
#            if read.is_reverse:
#                fragmentStart = read.aend - defaultFragmentLength
#                fragmentEnd = read.aend
#            else:
#                fragmentStart = read.pos
#                fragmentEnd = read.pos + defaultFragmentLength
#
#    return fragmentStart, fragmentEnd


def run_shell_command(command):
    """
    Runs the given shell command. Report
    any errors found.
    """
    try:
        subprocess.check_call(command, shell=True)

    except subprocess.CalledProcessError as error:
        logging.error(f'Error{error}\n')
        exit(1)
    except Exception as error:
        logging.error(f'Error: {error}\n')
        exit(1)


def main(args=None):
    global verbose_flag, F_gc, N_gc, R_gc, global_vars

    args = process_args(args)
    verbose_flag = args.verbose

    loglevel = logging.INFO
    logformat = '%(message)s'
    if args.verbose:
        loglevel = logging.DEBUG
        logformat = "%(asctime)s: %(levelname)s - %(message)s"

    logging.basicConfig(stream=sys.stderr, level=loglevel, format=logformat)
    # data = np.loadtxt(args.GCbiasFrequenciesFile.name)
    data = pd.read_csv(args.GCbiasFrequenciesFile.name, sep="\t", index_col=[0, 1])

    F_gc = data.loc["F_gc"]
    N_gc = data.loc["N_gc"]
    R_gc = data.loc["R_gc"]

    N_GC_min, N_GC_max = np.nanmin(N_gc.index), np.nanmax(N_gc.index)

    global_vars = dict()
    global_vars['2bit'] = args.genome
    global_vars['bam'] = args.bamfile

    # compute the probability to find more than one read (a redundant read)
    # at a certain position based on the gc of the read fragment
    # the binomial function is used for that
    # max_dup_gc = [binom.isf(1e-7, F_gc[x], 1.0 / N_gc[x])
    #              if F_gc[x] > 0 and N_gc[x] > 0 else 1
    #              for x in range(len(F_gc))]
    max_dup_gc = dict()
    for i in np.arange(N_GC_min, N_GC_max + 1, 1):
        N_tmp = N_gc.loc[i].to_numpy()
        F_tmp = F_gc.loc[i].to_numpy()
        max_dup_gc[i] = [binom.isf(1e-7, F_tmp[x], 1.0 / N_tmp[x])
                         if F_tmp[x] > 0 and N_tmp[x] > 0 else 1
                         for x in range(len(F_tmp))]
    if verbose_flag:
        logging.debug(f"max_dup_gc: {max_dup_gc}")
    global_vars['max_dup_gc'] = max_dup_gc

    tbit = py2bit.open(global_vars['2bit'])
    bam, mapped, unmapped, stats = openBam(args.bamfile, returnStats=True, nThreads=args.numberOfProcessors)

    global_vars['genome_size'] = sum(tbit.chroms().values())
    global_vars['total_reads'] = mapped
    global_vars['reads_per_bp'] = \
        float(global_vars['total_reads']) / args.effectiveGenomeSize

    # apply correction
    logging.info("applying correction")
    # divide the genome in fragments containing about 4e5 reads.
    # This amount of reads takes about 20 seconds
    # to process per core (48 cores, 256 Gb memory)
    chunkSize = int(4e5 / global_vars['reads_per_bp'])

    # chromSizes: list of tuples
    chromSizes = [(bam.references[i], bam.lengths[i])
                  for i in range(len(bam.references))]

    regionStart = 0
    if args.region:
        chromSizes, regionStart, regionEnd, chunkSize = mapReduce.getUserRegion(chromSizes, args.region,
                                                                                max_chunk_size=chunkSize)

    logging.info(f"genome partition size for multiprocessing: {chunkSize}")
    logging.info(f"using region {args.region}")
    mp_args = []

    chrNameBitToBam = tbitToBamChrName(list(tbit.chroms().keys()), bam.references)
    chrNameBamToBit = dict([(v, k) for k, v in chrNameBitToBam.items()])
    logging.info(f"{chrNameBitToBam}, {chrNameBamToBit}")
    c = 1
    pool = multiprocessing.Pool(args.numberOfProcessors)
    if args.correctedFile.name.endswith('bam'):
        for chrom, size in chromSizes:
            start = 0 if regionStart == 0 else regionStart
            for i in range(start, size, chunkSize):
                try:
                    chrNameBamToBit[chrom]
                except KeyError:
                    logging.debug(f"no sequence information for chromosome {chrom} in 2bit file")
                    logging.debug("Reads in this chromosome will be skipped")
                    continue
                chunk_end = min(size, i + chunkSize)
                mp_args.append((chrom, chrNameBamToBit[chrom], i, chunk_end, args.weight_only, verbose_flag))
                c += 1
        if len(mp_args) > 1 and args.numberOfProcessors > 1:
            logging.info(f"using {args.numberOfProcessors} processors for {len(mp_args)} number of tasks")

            res = pool.map_async(writeCorrectedSam_wrapper, mp_args).get(9999999)
        else:
            res = list(map(writeCorrectedSam_wrapper, mp_args))

        if len(res) == 1:
            command = f"cp {res[0]} {args.correctedFile.name}"
            run_shell_command(command)
        else:
            logging.info("concatenating (sorted) intermediate BAMs")
            header = pysam.Samfile(res[0])
            of = pysam.Samfile(args.correctedFile.name, "wb", template=header)
            header.close()
            for f in res:
                f = pysam.Samfile(f)
                for e in f.fetch(until_eof=True):
                    of.write(e)
                f.close()
            of.close()

        logging.info("indexing BAM")
        pysam.index(args.correctedFile.name)  # usable through pysam dispatcher

        for tempFileName in res:
            os.remove(tempFileName)

    if args.correctedFile.name.endswith('bg') or args.correctedFile.name.endswith('bw'):
        bedGraphStep = args.binSize  # 50 per default
        for chrom, size in chromSizes:
            start = 0 if regionStart == 0 else regionStart
            for i in range(start, size, chunkSize):
                try:
                    chrNameBamToBit[chrom]
                except KeyError:
                    logging.debug(f"no sequence information for chromosome {chrom} in 2bit file")
                    logging.debug("Reads in this chromosome will be skipped")
                    continue
                segment_end = min(size, i + bedGraphStep)
                mp_args.append((chrom, chrNameBamToBit[chrom], i, segment_end, bedGraphStep))
                c += 1

        if len(mp_args) > 1 and args.numberOfProcessors > 1:
            res = pool.map_async(writeCorrected_wrapper, mp_args).get(9999999)
        else:
            res = list(map(writeCorrected_wrapper, mp_args))

        oname = args.correctedFile.name
        args.correctedFile.close()
        if oname.endswith('bg'):
            f = open(oname, 'wb')
            for tempFileName in res:
                if tempFileName:
                    shutil.copyfileobj(open(tempFileName, 'rb'), f)
                    os.remove(tempFileName)
            f.close()
        else:
            chromSizes = [(k, v) for k, v in tbit.chroms().items()]
            writeBedGraph.bedGraphToBigWig(chromSizes, res, oname)


class Tester:
    def __init__(self):
        global debug, global_vars
        self.root = os.path.dirname(os.path.abspath(__file__)) + "/test/test_corrGC/"
        self.tbitFile = self.root + "sequence.2bit"
        self.bamFile = self.root + "test.bam"
        self.chrNameBam = '2L'
        self.chrNameBit = 'chr2L'
        bam, mapped, unmapped, stats = openBam(self.bamFile, returnStats=True)
        tbit = py2bit.open(self.tbitFile)
        debug = 0
        global_vars = {'2bit': self.tbitFile,
                       'bam': self.bamFile,
                       'filter_out': None,
                       'extra_sampling_file': None,
                       'max_reads': 5,
                       'min_reads': 0,
                       'reads_per_bp': 0.3,
                       'total_reads': mapped,
                       'genome_size': sum(tbit.chroms().values())}

    def testWriteCorrectedChunk(self):
        """ prepare arguments for test
        """
        global R_gc, R_gc_min, R_gc_max
        R_gc = np.loadtxt(self.root + "R_gc_paired.txt")
        global_vars['max_dup_gc'] = np.ones(301)
        start = 200
        end = 300
        bedGraphStep = 25
        return (self.chrNameBam,
                self.chrNameBit, start, end, bedGraphStep)

    def testWriteCorrectedSam(self):
        """ prepare arguments for test
        """
        global R_gc, R_gc_min, R_gc_max
        R_gc = np.loadtxt(self.root + "R_gc_paired.txt")
        global_vars['max_dup_gc'] = np.ones(301)
        start = 200
        end = 250
        return (self.chrNameBam,
                self.chrNameBit, start, end)

    def testWriteCorrectedSam_paired(self):
        """ prepare arguments for test.
        """
        global R_gc, R_gc_min, R_gc_max, global_vars
        R_gc = np.loadtxt(self.root + "R_gc_paired.txt")
        start = 0
        end = 500
        global_vars['bam'] = self.root + "paired.bam"
        return 'chr2L', 'chr2L', start, end


if __name__ == "__main__":
    main()
