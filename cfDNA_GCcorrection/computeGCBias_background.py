#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import multiprocessing
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from itertools import starmap

import click
import numpy as np
import pandas as pd

# from cfDNA_GCcorrection.parserCommon import output
import py2bit
import pybedtools as pbt
from csaps import csaps
from loguru import logger
from mpire import WorkerPool
from mpire.utils import chunk_tasks
from scipy.stats import poisson

from cfDNA_GCcorrection import bamHandler
from cfDNA_GCcorrection.utilities import (
    getGC_content,
    hash_file,
    map_chroms,
    write_precomputed_table,
)

###### Set constants ######

STANDARD_CHROMOSOMES = (
    [str(i) for i in range(1, 23)]
    + ["X", "Y"]
    + ["chr" + str(i) for i in range(1, 23)]
    + ["chrX", "chrY"]
)


###### define functions doing the work ######
def flatten(xs):
    for x in xs:
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def get_regions(
    genome, bam, nregions=10, windowsize=1000, blacklist=None, region=None, seed=None
):
    logger.info("Generating random regions...")
    if region:
        region_lst = region.replace("-", ":").split(":")
        if region_lst[0] in bam.references:
            reg_dict = {region_lst[0]: (int(region_lst[1]), int(region_lst[2]))}
            reg_coord = region.replace("-", ":").replace(":", " ")
        elif "chr" + region_lst[0] in bam.references:
            reg_dict = {"chr" + region_lst[0]: (int(region_lst[1]), int(region_lst[2]))}
            reg_coord = "chr" + region.replace("-", ":").replace(":", " ")
        elif region_lst[0].removeprefix("chr") in bam.references:
            reg_dict = {
                region_lst[0].removeprefix("chr"): (
                    int(region_lst[1]),
                    int(region_lst[2]),
                )
            }
            reg_coord = region.removeprefix("chr").replace("-", ":").replace(":", " ")
        else:
            raise Exception(
                f"ERROR: specified region {region} is not in:\n{bam.references}"
            )
        reg_filter = pbt.BedTool(reg_coord, from_string=True).saveas()
        if seed:
            random_regions = (
                pbt.BedTool()
                .random(l=windowsize, n=nregions, g=reg_dict, seed=seed)
                .shuffle(g=reg_dict, incl=reg_filter.fn, noOverlapping=True, seed=seed)
                .tabix()
            )
        else:
            random_regions = (
                pbt.BedTool()
                .random(l=windowsize, n=nregions, g=reg_dict)
                .shuffle(g=reg_dict, incl=reg_filter.fn, noOverlapping=True)
                .tabix()
            )
    else:
        if seed:
            random_regions = (
                pbt.BedTool()
                .random(l=windowsize, n=nregions, g=genome, seed=seed)
                .tabix()
            )
        else:
            random_regions = (
                pbt.BedTool().random(l=windowsize, n=nregions, g=genome).tabix()
            )
    if blacklist:
        logger.info("Removing blacklisted regions.")
        blacklist_df = pbt.BedTool(blacklist).to_dataframe()
        blacklist_map = map_chroms(set(blacklist_df["chrom"].unique()), genome)
        mapped_blacklist_df = blacklist_df.replace({"chrom": blacklist_map})
        mapped_blacklist = pbt.BedTool().from_dataframe(
            mapped_blacklist_df.loc[
                mapped_blacklist_df["chrom"].isin(blacklist_map.values())
            ]
        )
        filtered_regions = random_regions.subtract(
            mapped_blacklist, A=True
        )  # A=True: Remove entire feature if any overlap.
        return (
            filtered_regions.to_dataframe(
                dtype={"chrom": "category", "start": "uint32", "end": "uint32"}
            )
            .drop(columns=["name", "score", "strand"])
            .to_dict(orient="records")
        )
    return (
        random_regions.to_dataframe(
            dtype={"chrom": "category", "start": "uint32", "end": "uint32"}
        )
        .drop(columns=["name", "score", "strand"])
        .to_dict(orient="records")
    )


def roundGCLenghtBias(gc):
    gc_frac, gc_int = math.modf(round(gc * 100, 2))
    gc_new = gc_int + rng.binomial(1, gc_frac)
    return int(gc_new)


def get_N_GC(
    chrom,
    start,
    end,
    reference,
    fragment_lengths,
    chr_name_bam_to_bit_mapping,
    steps=1,
    verbose=False,
):
    sub_Ndict = dict()
    tbit_chrom = chr_name_bam_to_bit_mapping[chrom]
    logger.debug(f"chrom: {chrom}; mapped tbit_chrom: {tbit_chrom}")
    for flen in fragment_lengths:
        sub_n_gc = np.zeros(100 + 1, dtype="int")
        for pos in range(start, end - flen + 1, steps):
            try:
                gc = getGC_content(
                    reference, tbit_chrom, pos, int(pos + flen), fraction=True
                )
                gc = roundGCLenghtBias(gc)
            except Exception as detail:
                if verbose:
                    logger.exception(detail)
                continue
            sub_n_gc[gc] += 1
        sub_Ndict[str(flen)] = sub_n_gc
    return sub_Ndict


###### worker definition for ray ######


def tabulateGCcontent_wrapper(param_dict, *chunk):
    logger.debug(f"Worker starting to work on chunk: {chunk}")
    logger.debug(f"param_dict: {param_dict}\n chunk: {chunk}")
    logger.debug("Setting up wrapper dictionary.")

    wrapper_ndict = dict()

    for task in flatten(chunk):
        logger.debug(f"Calculating values for task: {task}")
        subN_gc = tabulateGCcontent_worker(**task, **param_dict)
        logger.debug(f"Updating wrapper dictionaries.")
        wrapper_ndict = {
            k: wrapper_ndict.get(k, 0) + subN_gc.get(k, np.zeros(100 + 1, dtype="int"))
            for k in set(wrapper_ndict) | set(subN_gc)
        }

    logger.debug("Returning dictionary from wrapper.")

    return wrapper_ndict


def tabulateGCcontent_worker(
    chrom,
    start,
    end,
    chr_name_bam_to_bit_mapping,
    stepSize=1,
    fragment_lengths=None,
    verbose=False,
):

    tbit = py2bit.open(global_vars["2bit"])
    bam = bamHandler.openBam(global_vars["bam"])

    # print(f"using fragment lengths: {fragment_lengths}")
    sub_Ndict = get_N_GC(
        chrom,
        start,
        end,
        reference=tbit,
        steps=stepSize,
        fragment_lengths=fragment_lengths,
        verbose=verbose,
        chr_name_bam_to_bit_mapping=chr_name_bam_to_bit_mapping,
    )
    return sub_Ndict


###### wrap all measurement functions in meta function ######


def tabulateGCcontent(
    num_cpus,
    regions,
    chr_name_bam_to_bit_mapping,
    stepSize=1,
    fragment_lengths=None,
    mp_type="MP",
    verbose=False,
):

    global global_vars

    param_dict = {
        "stepSize": stepSize,
        "fragment_lengths": fragment_lengths,
        "chr_name_bam_to_bit_mapping": chr_name_bam_to_bit_mapping,
        "verbose": verbose,
    }

    if mp_type.lower() == "mp":

        TASKS = regions
        if len(TASKS) > 1 and num_cpus > 1:
            logger.info("Using python multiprocessing!")
            logger.info(
                ("Using {} processors for {} " "tasks".format(num_cpus, len(TASKS)))
            )
            chunked_tasks = chunk_tasks(TASKS, n_splits=num_cpus * 2)
            starmap_generator = ((param_dict, chunk) for chunk in chunked_tasks)
            pool = multiprocessing.Pool(num_cpus)
            imap_res = pool.starmap_async(
                tabulateGCcontent_wrapper, starmap_generator, chunksize=1
            ).get(9999999)
            pool.close()
            pool.join()
        else:
            starmap_generator = ((param_dict, chunk) for chunk in TASKS)
            imap_res = starmap(tabulateGCcontent_wrapper, starmap_generator)
    elif mp_type.lower() == "mpire":
        TASKS = regions
        if len(TASKS) > 1 and num_cpus > 1:
            logger.info("Using mpire multiprocessing!")
            logger.info(
                (
                    "Using {} processors for {} "
                    "number of tasks".format(num_cpus, len(TASKS))
                )
            )
            chunked_tasks = chunk_tasks(TASKS, n_splits=num_cpus * 2)
            with WorkerPool(n_jobs=num_cpus, shared_objects=param_dict) as pool:
                imap_res = pool.imap_unordered(
                    tabulateGCcontent_wrapper, chunked_tasks, chunk_size=1
                )
        else:
            starmap_generator = ((param_dict, chunk) for chunk in TASKS)
            imap_res = starmap(tabulateGCcontent_wrapper, starmap_generator)

    ndict = {
        str(key): np.zeros(100 + 1, dtype="int") for key in fragment_lengths
    }  # dict()

    for subN_gc in imap_res:
        ndict = {
            k: ndict.get(k, 0) + subN_gc.get(k, 0) for k in set(ndict) | set(subN_gc)
        }
        # fdict = {k: fdict.get(k, 0) + subF_gc.get(k, 0) for k in set(fdict) | set(subF_gc)}

    # create multi-index dict
    data_dict = {
        "N_gc": ndict,
    }
    multi_index_dict = {
        (i, j): data_dict[i][j] for i in data_dict.keys() for j in data_dict[i].keys()
    }
    data = pd.DataFrame.from_dict(multi_index_dict, orient="index")
    data.index = pd.MultiIndex.from_tuples(data.index)
    data.index = data.index.set_levels(
        data.index.levels[-1].astype(int), level=-1
    )  # set length index to integer for proper sorting
    data.sort_index(inplace=True)

    # filter data for standard values (all zero), except for first and last column
    Ndata = data.loc["N_gc"]
    Ndata_filtered = Ndata.loc[
        Ndata.index.isin(Ndata.index[[0, -1]]) | (Ndata != 0).any(axis=1)
    ]
    Ndata_multiindex = pd.concat({"N_gc": Ndata_filtered})

    return Ndata_multiindex


###### This part is the command line interface and main function ######


@click.command()
@click.option(
    "--bamfile",
    "-b",
    "bamfile",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Sorted BAM file.",
)
@click.option(
    "--genome",
    "-g",
    "genome",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="""Genome in two bit format. Most genomes can be 
            found here: http://hgdownload.cse.ucsc.edu/gbdb/ 
            Search for the .2bit ending. Otherwise, fasta 
            files can be converted to 2bit using the UCSC 
            programm called faToTwoBit available for different 
            plattforms at '
            http://hgdownload.cse.ucsc.edu/admin/exe/""",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    required=True,
    type=click.Path(writable=True),
    help="""Path to save the file containing 
            the expected read frequencies per %%GC-
            content. This file can be provided as precomputed 
            background to the computeGCBias_readlen script. 
            This is a tab separated file.""",
)
@click.option(
    "--num_cpus",
    "-p",
    "num_cpus",
    type=click.INT,
    default=1,
    show_default=True,
    help="Number of processors to use."
    # Type "max/2" to use half the maximum number of processors or "max" to use all available processors",  <- need implementation!
)
@click.option(
    "--minlen",
    "-min",
    "minlen",
    default=30,
    show_default=True,
    type=click.INT,
    help="Minimum fragment length to consider for bias computation.",
)
@click.option(
    "--maxlen",
    "-max",
    "maxlen",
    default=250,
    show_default=True,
    type=click.INT,
    help="Maximum fragment length to consider for bias computation.",
)
@click.option(
    "--lengthstep",
    "-fstep",
    "lengthStep",
    default=1,
    # show_default=True,
    type=click.INT,
    help="""Step size for fragment lenghts between minimum and maximum fragment length.
            Will be ignored if interpolate is deactivated.""",
)
@click.option(
    "--interpolate",
    "-i",
    "interpolate",
    is_flag=True,
    default=False,
    show_default=True,
    help="""Interpolates GC values and correction for missing read lengths.
            This might substantially reduce computation time, but might lead to
            less accurate results. Deactivated by default.""",
)
@click.option(
    "--sampleSize",
    "sampleSize",
    default=5e7,
    show_default=True,
    type=click.INT,
    help="""Number of sampling points to be considered. Will be filtered if blacklist file is provided.""",
)
@click.option(
    "--blacklistfile",
    "-bl",
    "blacklistfile",
    type=click.Path(exists=True, readable=True),
    help="""A BED file containing regions that should be excluded from all analyses.
            Currently this works by rejecting genomic chunks that happen to overlap an entry.""",
)
@click.option(
    "-sc",
    "--standard_chroms",
    "standard_chroms",
    is_flag=True,
    help="Flag: filter chromosomes to human standard chromosomes.",
)
@click.option(
    "--region",
    "-r",
    "region",
    type=click.STRING,
    help="""Region of the genome to limit the operation 
     to - this is useful when testing parameters to 
     reduce the computing time. The format is 
    chr:start-end, for example 
    --region chr10:456700-891000 or
    --region 10:456700-891000.""",
)
@click.option(
    "--seed",
    "seed",
    default=None,
    type=click.INT,
    help="""Set seed for reproducibility.""",
)
@click.option(
    "--mp_backend",
    "-mp",
    "mp_type",
    type=click.Choice(["MP", "MPIRE"], case_sensitive=False),
    default="MPIRE",
    show_default=True,
    help="Specifies the multiprocessing backend. MP = python multiprocessing, MPIRE for using mpire.",
)
@click.option("-v", "--verbose", "verbose", is_flag=True, help="Enables verbose mode")
@click.option("--debug", "debug", is_flag=True, help="Enables debug mode")
def main(
    bamfile,
    genome,
    output_file,
    num_cpus,
    minlen,
    maxlen,
    lengthStep,
    interpolate,
    sampleSize,
    blacklistfile,
    region,
    standard_chroms,
    verbose,
    debug,
    seed,
    mp_type,
):

    if debug:
        debug_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>process: {process}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
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

    logger.info("Setting up variables")
    # set a global random number generator for numpy
    global rng
    rng = np.random.default_rng(seed=seed)
    random.seed(seed)

    global global_vars
    global_vars = dict()
    global_vars["2bit"] = genome
    global_vars["bam"] = bamfile

    tbit = py2bit.open(global_vars["2bit"])
    bam, mapped, unmapped, stats = bamHandler.openBam(
        global_vars["bam"], returnStats=True, nThreads=num_cpus
    )
    if interpolate:
        # currently needs to be explicitly set -> should be 1, a length filter for F_GC needs implementation
        length_step = lengthStep
    else:
        length_step = 1

    fragment_lengths = np.arange(minlen, maxlen + 1, length_step).tolist()

    for key in global_vars:
        logger.debug(f"{key}: {global_vars[key]}")

    # the chromosome names in chrom_dict/regions will be based on the bam file. For accessing tbit files, a mapping is needed.
    chr_name_bam_to_bit_mapping = map_chroms(
        bam.references,
        list(tbit.chroms().keys()),
        ref_name="bam file",
        target_name="2bit reference file",
    )

    if standard_chroms:
        # get valid chromosomes and their lenght as dict that can be used by pybedtools and filter for human standard chromosomes.
        # This represents the information in a .genome file.
        chrom_dict = {
            bam.references[i]: (0, bam.lengths[i])
            for i in range(len(bam.references))
            if bam.references[i] in STANDARD_CHROMOSOMES
            and chr_name_bam_to_bit_mapping.keys()
        }
    else:
        chrom_dict = {
            bam.references[i]: (0, bam.lengths[i])
            for i in range(len(bam.references))
            if bam.references[i] in chr_name_bam_to_bit_mapping.keys()
        }

    regions = get_regions(
        genome=chrom_dict,
        bam=bam,
        nregions=sampleSize,
        windowsize=1000,
        blacklist=blacklistfile,
        region=region,
        seed=seed,
    )

    sampleSize_regions = sampleSize / 1000
    regions = random.sample(regions, sampleSize_regions)

    logger.debug(f"regions contains {len(regions)} genomic coordinates")
    # logger.info("computing frequencies")
    logger.info("Computing frequencies...")
    # the GC of the genome is sampled each stepSize bp.
    # step_size = max(int(global_vars["genome_size"] / sampleSize), 1)
    # logger.info(f"stepSize for genome sampling: {step_size}")

    data = tabulateGCcontent(
        chr_name_bam_to_bit_mapping=chr_name_bam_to_bit_mapping,
        num_cpus=num_cpus,
        regions=regions,
        fragment_lengths=fragment_lengths,
        mp_type=mp_type,
    )

    region_params = {
        "genome": chrom_dict,
        "nregions": sampleSize,
        "windowsize": 1000,
        "region": region,
        "seed": seed,
    }
    if blacklistfile:
        blacklist_hash = hash_file(blacklistfile)
        out_dict = {
            "blacklist_hash": blacklist_hash,
            "get_regions_params": region_params,
        }
    else:
        out_dict = {"blacklist_hash": None, "get_regions_params": region_params}

    write_precomputed_table(df=data, params_dict=out_dict, filename=output_file)


if __name__ == "__main__":
    main()
