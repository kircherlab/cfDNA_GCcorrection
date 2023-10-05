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

    if (
        read.is_paired and read.is_proper_pair
    ):  # and abs(read.tlen) < 2 * fragmentLength:
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


@click.command()
@click.option(
    "--bamfile",
    "-b",
    "bam_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Sorted BAM file.",
)
@click.option(
    "--genome",
    "-g",
    "reference_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="""Genome reference in two bit format. Most genomes can be 
            found here: http://hgdownload.cse.ucsc.edu/gbdb/ 
            Search for the .2bit ending. Otherwise, fasta 
            files can be converted to 2bit using the UCSC 
            programm called faToTwoBit available for different 
            plattforms at '
            http://hgdownload.cse.ucsc.edu/admin/exe/""",
)
@click.option(
    "--GCbiasFrequenciesFile",
    "-freq",
    "GCbias_frequencies_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="""Indicate the output file from computeGCBias containing 
            the coverage bias values per GC-content and fragment length.
            This file should be a (gzipped) tab separated file.""",
)
@click.option(
    "--outfile",
    "-o",
    "output_file",
    required=True,
    type=click.Path(writable=True),
    help="""Output Bam file""",
)
@click.option(
    "--num_cpus",
    "-p",
    "num_cpus",
    type=click.INT,
    default=1,
    show_default=True,
    help="Number of processors to use.",
)
@click.option(
    "--weights/--copies",
    "weight_only",
    type=click.BOOL,
    default=True,
    show_default=True,
    help="""Flag for either attaching weights or 
            altering copies of reads based on GC content.""",
)
@click.option(
    "--region",
    "region",
    type=click.STRING,
    default=None,
    show_default=True,
    help="""Genomic region specified for correcting reads (e.g chr1:0-1000).""",
)
@click.option(
    "--effectiveGenomeSize",
    "effective_genome_size",
    type=click.INT,
    default=None,
    show_default=True,
    help="""[Optional]:
            Effective genome size, used to tune window sizes for paralellisation.
            
            The effective genome size is the portion 
            of the genome that is mappable. Large fractions of 
            the genome are stretches of NNNN that should be 
            discarded. Also, if repetitive regions were not 
            included in the mapping of reads, the effective 
            genome size needs to be adjusted accordingly. 
            A table of values is available here: 
            http://deeptools.readthedocs.io/en/latest/content/feature/effectiveGenomeSize.html""",
)
@click.option(
    "--use_nearest_weight",
    "use_nearest_weight",
    is_flag=True,
    default=False,
    show_default=True,
    help="""Use nearest weight, if a fragment length is not included in the GC profile.
            This should be only used with length filtered bam files or weights might
            be not representative of fragments!""",
)
@click.option(
    "--num_threads",
    "pysam_compression_threads",
    type=click.INT,
    default=10,
    show_default=True,
    help="Number of compression threads used for BAM I/O.",
)
@click.option(
    "--default_weight",
    "default_value",
    type=click.INT,
    default=1,
    show_default=True,
    help="Default weight for fragment lengths not included in GC profile.",
)
@click.option(
    "--seed",
    "seed",
    default=None,
    type=click.INT,
    help="""Set seed for reproducibility.""",
)
@click.option("--progress_bar", "progress_bar", is_flag=True, help="Enables TQDM progress bar.")
@click.option("-v", "--verbose", "verbose", is_flag=True, help="Enables verbose mode.")
@click.option("--debug", "debug", is_flag=True, help="Enables debug mode.")
def main(
    bam_file,
    reference_file,
    GCbias_frequencies_file,
    output_file,
    num_cpus=1,
    weight_only=True,
    region=None,
    effective_genome_size=None,
    use_nearest_weight=False,
    pysam_compression_threads=10,
    default_value=1,
    seed=None,
    progress_bar=False,
    verbose=False,
    debug=False,
):
    ### initial setup

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

    global rng
    rng = np.random.default_rng(seed=seed)
    random.seed(seed)

    logger.debug("Provided arguments:")
    logger.debug(locals())

    start_time = time.time()

    ## set parameters
    logger.info("Preparing parameters.")

    ### load and process GC bias profile
    logger.info("Loading GC bias profile.")
    data = pd.read_csv(GCbias_frequencies_file, sep="\t", index_col=[0, 1])

    F_gc = data.loc["F_gc"]  # fragment counts
    N_gc = data.loc["N_gc"]  # genomic background
    R_gc = data.loc["R_gc"]  # bias weights

    logger.debug(R_gc.T.describe().T.describe())

    R_gc.columns = R_gc.columns.astype(int)
    R_gc_dict = R_gc.rdiv(1).round(2).to_dict(orient="index")

    N_GC_min, N_GC_max = np.nanmin(N_gc.index), np.nanmax(N_gc.index)

    ### estimate max duplication based on the GC profile if reads
    ### should be duplicated instead of attaching weights

    max_dup_gc = None
    if not weight_only:
        # compute the probability to find more than one read (a redundant read)
        # at a certain position based on the gc of the read fragment
        # the binomial function is used for that
        # max_dup_gc = [binom.isf(1e-7, F_gc[x], 1.0 / N_gc[x])
        #              if F_gc[x] > 0 and N_gc[x] > 0 else 1
        #              for x in range(len(F_gc))]
        logger.info("Estimate the probability of redundant reads based on GC profile.")
        max_dup_gc = dict()
        for i in np.arange(N_GC_min, N_GC_max + 1, 1):
            N_tmp = N_gc.loc[i].to_numpy()
            F_tmp = F_gc.loc[i].to_numpy()
            max_dup_gc[i] = [
                binom.isf(1e-7, F_tmp[x], 1.0 / N_tmp[x])
                if F_tmp[x] > 0 and N_tmp[x] > 0
                else 1
                for x in range(len(F_tmp))
            ]

        logger.debug(f"max_dup_gc: {max_dup_gc}")

    ### get bam stats
    logger.info("Loading genome and bam file.")
    tbit = py2bit.open(reference_file)
    bam, mapped, unmapped, bam_stats = openBam(
        bam_file, returnStats=True, nThreads=num_cpus
    )

    genome_size = sum(tbit.chroms().values())
    total_reads = mapped

    if effective_genome_size:
        reads_per_bp = float(total_reads) / effective_genome_size
    else:
        reads_per_bp = float(total_reads) / genome_size

    logger.debug(
        f"Bam stats: mapped reads: {mapped}; unmapped reads: {unmapped}; \
            genome size: {genome_size}; estimated reads per bp: {reads_per_bp}"
    )

    ### preparing chunks for parallel processing

    logger.info("Preparing chunks for processing.")
    # divide the genome in fragments containing about 4e5 reads.
    # This amount of reads takes about 20 seconds
    # to process per core (48 cores, 256 Gb memory)
    chunk_size = int(4e5 / reads_per_bp)

    # chrom_sizes: list of tuples
    chrom_sizes = [
        (bam.references[i], bam.lengths[i]) for i in range(len(bam.references))
    ]

    region_start = 0

    if region:
        logger.info(
            f"Using user defined region {region} for correction. \
            Other regions will not be corrected!"
        )
        region_cleaned = region.replace("-", ":")
        chrom_sizes, region_start, region_end, chunk_size = getUserRegion(
            chrom_sizes, region_cleaned, max_chunk_size=chunk_size
        )

    chr_name_bam_to_bit_mapping = map_chroms(
        bam.references,
        list(tbit.chroms().keys()),
        ref_name="bam file",
        target_name="2bit reference file",
    )

    chunks = get_chunks(
        chrom_sizes=chrom_sizes,
        region_start=region_start,
        chunk_size=chunk_size,
        chr_name_bam_to_bit_mapping=chr_name_bam_to_bit_mapping,
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

    ### preparing shared parameters

    logger.info("Preparing shared objects.")

    shared_params = {
        "bam_file": bam_file,
        "twobit_file": reference_file,
        "R_gc_dict": R_gc_dict,
        "max_dup_gc": max_dup_gc,
        "tag_but_not_change_number": weight_only,
        "verbose": verbose,
        "debug": debug,
        "threads": pysam_compression_threads,
        "default_value": default_value,
        "use_nearest_weight": use_nearest_weight,
    }

    ## do the computation

    logger.info("Starting correction.")

    if len(chunks) > 1 and num_cpus > 1:
        logger.info(
            f"Using python multiprocessing with {(num_cpus-1)} CPU cores for {len(chunks)} tasks"
        )
        with WorkerPool(n_jobs=(num_cpus - 1), shared_objects=shared_params) as pool:
            imap_res = pool.imap(
                writeCorrectedBam_wrapper,
                make_single_arguments(chunks),
                iterable_len=len(chunks),
                progress_bar=progress_bar,
            )
    else:
        logger.info(f"Using one process for for {len(chunks)} tasks")
        starmap_generator = ((shared_params, chunk) for chunk in chunks)
        imap_res = starmap(writeCorrectedBam_wrapper, starmap_generator)

    ## aggregate results

    if len(chunks) == 1:
        res = list(imap_res)
        command = f"cp {res[0]} {output_file}"
        run_shell_command(command)
    else:
        logger.info("Concatenating (sorted) intermediate BAMs")
        out_threads = math.ceil(pysam_compression_threads * 2 / 3)
        in_threads = max(1, (pysam_compression_threads - out_threads))

        res = list()
        pysam_verbosity = pysam.set_verbosity(0) # set htslib error verbosity to 0, as we expect the tmpfiles not to have an index  # noqa: E501

        with pysam.AlignmentFile(
            output_file, "wb", template=bam, threads=out_threads
        ) as of:
            for tmpfile in imap_res:
                res.append(tmpfile)
                logger.info(f"Adding tmpfile({tmpfile}) to final output file.")
                with pysam.AlignmentFile(tmpfile, "rb", threads=in_threads) as file:
                    for read in file.fetch(until_eof=True):
                        of.write(read)
                if not progress_bar:
                    elapsed_time = time.time() - start_time
                    logger.info(f"Progress: {len(res)}/{len(chunks)} tasks completed in {int(elapsed_time/3600)}:{int(elapsed_time%3600/60):02d}:{int(elapsed_time%60):02d} (HH:MM:SS).")  # noqa: E501
        
        pysam.set_verbosity(pysam_verbosity) # reset htslib verbosity to previous state
        logger.info(f"Indexing BAM: {output_file}")
        pysam.index(output_file)  # usable through pysam dispatcher

        logger.info("Removing temporary files.")
        for tempFileName in res:
            os.remove(tempFileName)

        elapsed_time = time.time() - start_time
        logger.info(f"Full computation took {int(elapsed_time/3600)}:{int(elapsed_time%3600/60):02d}:{int(elapsed_time%60):02d} (HH:MM:SS).")

if __name__ == "__main__":
    main(max_content_width=120)
