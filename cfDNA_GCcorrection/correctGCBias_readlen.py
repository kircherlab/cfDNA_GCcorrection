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


def get_chunks(chrom_sizes, region_start, chunk_size, chr_name_bam_to_bit_mapping):
    chunks = list()
    for chrom, size in chrom_sizes:
        for i in range(region_start, size, chunk_size):
            try:
                chrNameBit = chr_name_bam_to_bit_mapping[chrom]
            except KeyError:
                logger.debug(
                    f"No sequence information for chromosome {chrom} in 2bit file. \
                        Reads in this chromosome will be skipped"
                )
                continue
            chunk_end = min(size, i + chunk_size)
            chunks.append(
                {
                    "chrNameBam": chrom,
                    "chrNameBit": chrNameBit,
                    "start": i,
                    "end": chunk_end,
                }
            )
    return chunks


def writeCorrectedBam_wrapper(shared_params, chunk):
    return writeCorrectedBam_worker(**shared_params, **chunk)


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


    if debug:
        debug_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | \
            <level>{level: <8}</level> | <level>process: {process}</level> | \
            <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - \
            <level>{message}</level>"
        logger.remove()
        logger.add(
            sys.stderr, level="DEBUG", format=debug_format, colorize=True, enqueue=True
        )
        logger.debug("Debug mode active.")
    elif verbose:
        info_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{message}</level>"
        )
        logger.remove()
        logger.add(
            sys.stderr, level="INFO", format=info_format, colorize=False, enqueue=True
        )
    else:
        logger.remove()
        logger.add(
            sys.stderr,
            level="WARNING",
            colorize=True,
            enqueue=True,
        )

    logger.debug("Provided arguments:")
    logger.debug(locals())

    logger.info("Preparing parameters.")
    logger.info("Loading GC bias profile.")
    logger.debug(R_gc.T.describe().T.describe())

    N_GC_min, N_GC_max = np.nanmin(N_gc.index), np.nanmax(N_gc.index)

    global_vars = dict()
    global_vars['2bit'] = args.genome
    global_vars['bam'] = args.bamfile

        logger.info("Estimate the probability of redundant reads based on GC profile.")
        max_dup_gc = dict()
        logger.debug(f"max_dup_gc: {max_dup_gc}")
    logger.info("Loading genome and bam file.")
    logger.debug(
        f"Bam stats: mapped reads: {mapped}; unmapped reads: {unmapped}; \
            genome size: {genome_size}; estimated reads per bp: {reads_per_bp}"
    )
    logger.info("Preparing chunks for processing.")
    # divide the genome in fragments containing about 4e5 reads.
    # This amount of reads takes about 20 seconds
    # to process per core (48 cores, 256 Gb memory)
    chunkSize = int(4e5 / global_vars['reads_per_bp'])

    # chromSizes: list of tuples
    chromSizes = [(bam.references[i], bam.lengths[i])
                  for i in range(len(bam.references))]

        logger.info(
            f"Using user defined region {region} for correction. \
            Other regions will not be corrected!"
        )
    # check if at least each CPU core gets a task
    if len(chunks) < (num_cpus - 1):
        logger.debug(f"Less chunks({len(chunks)}) than CPUs({(num_cpus-1)}) detected.")
        chunk_size = math.ceil(
            chunk_size * (len(chunks) / (num_cpus - 1))
        )  # adjust chunk_size so that each CPU core gets a task
        logger.debug(f"New chunk_size: {chunk_size}")
        chunks = get_chunks(
            chrom_sizes=chrom_sizes,
            region_start=region_start,
            chunk_size=chunk_size,
            chr_name_bam_to_bit_mapping=chr_name_bam_to_bit_mapping,
        )

    logger.info(f"Genome partition size for multiprocessing: {chunk_size}")

    logger.info("Preparing shared objects.")
    logger.info("Starting correction.")
        logger.info(
            f"Using python multiprocessing with {(num_cpus-1)} CPU cores for {len(chunks)} tasks"
        )
        logger.info(f"Using one process for for {len(chunks)} tasks")
        starmap_generator = ((shared_params, chunk) for chunk in chunks)
        logger.info("Concatenating (sorted) intermediate BAMs")
        out_threads = math.ceil(pysam_compression_threads * 2 / 3)
                logger.info(f"Adding tmpfile({tmpfile}) to final output file.")
                    logger.info(f"Progress: {len(res)}/{len(chunks)} tasks completed in {int(elapsed_time/3600)}:{int(elapsed_time%3600/60):02d}:{int(elapsed_time%60):02d} (HH:MM:SS).")  # noqa: E501
        logger.info(f"Indexing BAM: {output_file}")
        logger.info("Removing temporary files.")
        for tempFileName in res:
            os.remove(tempFileName)

        logger.info(f"Full computation took {int(elapsed_time/3600)}:{int(elapsed_time%3600/60):02d}:{int(elapsed_time%60):02d} (HH:MM:SS).")

if __name__ == "__main__":
    main()
