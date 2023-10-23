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
    map_chroms,
    hash_file,
    read_precomputed_table,
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


def get_F_GC(
    chrom, start, end, bam, reference, chr_name_bam_to_bit_mapping, verbose=False
):
    sub_Fdict = defaultdict(lambda: np.zeros(100 + 1, dtype="int"))
    tbit_chrom = chr_name_bam_to_bit_mapping[chrom]
    logger.debug(f"chrom: {chrom}; mapped tbit_chrom: {tbit_chrom}")
    for read in bam.fetch(chrom, start, end):
        r_len = 0
        if (
            read.reference_start >= start
            and read.reference_end is not None
            and read.reference_end <= end
        ):  # check for none, some reads are faulty and return None!
            if (
                read.is_proper_pair and read.next_reference_start > read.reference_start
            ):  # is proper pair and only counted once! is this the same as read.mate_is_reverse?
                r_len = abs(read.template_length)
                try:
                    gc = getGC_content(
                        reference,
                        tbit_chrom,
                        read.reference_start,
                        read.reference_end,
                        fraction=True,
                    )
                    gc = roundGCLenghtBias(gc)
                except Exception as detail:
                    if verbose:
                        logger.exception(detail)
                    continue
                sub_Fdict[str(r_len)][gc] += 1
            elif not read.is_paired:
                r_len = read.query_length
                try:
                    gc = getGC_content(
                        reference,
                        tbit_chrom,
                        read.reference_start,
                        read.reference_end,
                        fraction=True,
                    )
                    gc = roundGCLenghtBias(gc)
                except Exception as detail:
                    if verbose:
                        logger.exception(detail)
                    continue
                sub_Fdict[str(r_len)][gc] += 1
            else:
                continue
        else:
            continue
    return dict(sub_Fdict)


###### worker definition for ray ######


def tabulateGCcontent_wrapper(param_dict, *chunk):
    logger.debug(f"Worker starting to work on chunk: {chunk}")
    logger.debug(f"param_dict: {param_dict}\n chunk: {chunk}")
    logger.debug("Setting up wrapper dictionaries.")

    F_GC_only = param_dict["F_GC_only"]
    fragment_lengths = list(map(str, param_dict["fragment_lengths"]))

    if F_GC_only:

        logger.debug(f"Calculating F_GC only.")
        wrapper_fdict = dict()

        for task in flatten(chunk):
            logger.debug(f"Calculating values for task: {task}")
            subF_gc = tabulateGCcontent_worker(**task, **param_dict)
            logger.debug(f"Updating wrapper dictionaries.")
            wrapper_fdict = {
                k: wrapper_fdict.get(k, 0)
                + subF_gc.get(k, np.zeros(100 + 1, dtype="int"))
                for k in set(wrapper_fdict) | set(fragment_lengths)
            }  # In this case, we use only read lengths that are in the defined fragment lengths, use subN_gc keys as proxy

        logger.debug("Returning dictionaries from wrapper.")

        return wrapper_fdict

    else:

        wrapper_ndict = dict()
        wrapper_fdict = dict()

        for task in flatten(chunk):
            logger.debug(f"Calculating values for task: {task}")
            subN_gc, subF_gc = tabulateGCcontent_worker(**task, **param_dict)
            logger.debug(f"Updating wrapper dictionaries.")
            wrapper_ndict = {
                k: wrapper_ndict.get(k, 0)
                + subN_gc.get(k, np.zeros(100 + 1, dtype="int"))
                for k in set(wrapper_ndict) | set(subN_gc)
            }
            wrapper_fdict = {
                k: wrapper_fdict.get(k, 0)
                + subF_gc.get(k, np.zeros(100 + 1, dtype="int"))
                for k in set(wrapper_fdict) | set(fragment_lengths)
            }  # In this case, we use only read lengths that are in the defined fragment lengths
        logger.debug("Returning dictionaries from wrapper.")

        return wrapper_ndict, wrapper_fdict


def tabulateGCcontent_worker(
    chrom,
    start,
    end,
    chr_name_bam_to_bit_mapping,
    stepSize=1,
    fragment_lengths=None,
    F_GC_only=False,
    verbose=False,
):

    tbit = py2bit.open(global_vars["2bit"])
    bam = bamHandler.openBam(global_vars["bam"])

    sub_Fdict = get_F_GC(
        chrom,
        start,
        end,
        bam=bam,
        reference=tbit,
        chr_name_bam_to_bit_mapping=chr_name_bam_to_bit_mapping,
    )
    if not F_GC_only:

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

        return sub_Ndict, sub_Fdict
    else:
        return sub_Fdict


###### wrap all measurement functions in meta function ######


def tabulateGCcontent(
    num_cpus,
    regions,
    chr_name_bam_to_bit_mapping,
    stepSize=1,
    fragment_lengths=None,
    Ndata=None,
    mp_type="MP",
    verbose=False,
):

    global global_vars

    if isinstance(Ndata, pd.DataFrame):
        F_GC_only = True
    else:
        F_GC_only = False

    param_dict = {
        "stepSize": stepSize,
        "fragment_lengths": fragment_lengths,
        "chr_name_bam_to_bit_mapping": chr_name_bam_to_bit_mapping,
        "verbose": verbose,
        "F_GC_only": F_GC_only,
    }

    # decide which of the multiprocessing backends to use
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

    ## decide whether to aggregate F_GC or N_GC and F_GC
    if F_GC_only:
        fdict = {str(key): np.zeros(100 + 1, dtype="int") for key in fragment_lengths}

        for subF_gc in imap_res:

            fdict = {
                k: fdict.get(k, 0) + subF_gc.get(k, 0) for k in set(fdict)
            }  # In this case, we use only read lengths that are in the defined fragment lengths

        # create multi-index dict
        data_dict = {"F_gc": fdict}

    else:

        ndict = {str(key): np.zeros(100 + 1, dtype="int") for key in fragment_lengths}
        fdict = {str(key): np.zeros(100 + 1, dtype="int") for key in fragment_lengths}

        for subN_gc, subF_gc in imap_res:
            ndict = {
                k: ndict.get(k, 0) + subN_gc.get(k, 0)
                for k in set(ndict) | set(subN_gc)
            }
            # fdict = {k: fdict.get(k, 0) + subF_gc.get(k, 0) for k in set(fdict) | set(subF_gc)}
            fdict = {
                k: fdict.get(k, 0) + subF_gc.get(k, 0) for k in set(fdict)
            }  # In this case, we use only read lengths that are in the defined fragment lengths

        # create multi-index dict
        data_dict = {"N_gc": ndict, "F_gc": fdict}

    multi_index_dict = {
        (i, j): data_dict[i][j] for i in data_dict.keys() for j in data_dict[i].keys()
    }
    data = pd.DataFrame.from_dict(multi_index_dict, orient="index")
    data.index = pd.MultiIndex.from_tuples(data.index)
    data.index = data.index.set_levels(
        data.index.levels[-1].astype(int), level=-1
    )  # set length index to integer for proper sorting
    data.sort_index(inplace=True)
    data.columns = data.columns.astype(str)

    # filter data for standard values (all zero), except for first and last column
    Fdata = data.loc["F_gc"]
    Fdata_filtered = Fdata.loc[
        Fdata.index.isin(Fdata.index[[0, -1]]) | (Fdata != 0).any(axis=1)
    ]
    Fdata_multiindex = pd.concat({"F_gc": Fdata_filtered})

    if F_GC_only:
        Ndata = Ndata.loc["N_gc"]
    else:
        Ndata = data.loc["N_gc"]

    Ndata_filtered = Ndata.loc[
        Ndata.index.isin(Ndata.index[[0, -1]]) | (Ndata != 0).any(axis=1)
    ]
    Ndata_multiindex = pd.concat({"N_gc": Ndata_filtered})
    filtered_data = pd.concat([Ndata_multiindex, Fdata_multiindex])

    return filtered_data


###### Processing mesured values either raw or with interpolation ######


def interpolate_ratio_csaps(df, smooth=None, normalized=False, minreads=4):
    # separate hypothetical read density from measured read density
    N_GC = df.loc["N_gc"]
    F_GC = df.loc["F_gc"]

    # get min and max values
    N_GC_min, N_GC_max = np.nanmin(N_GC.index.astype("int")), np.nanmax(
        N_GC.index.astype("int")
    )
    F_GC_min, F_GC_max = np.nanmin(F_GC.index.astype("int")), np.nanmax(
        F_GC.index.astype("int")
    )

    # sparse grid for hypothetical read density
    N_GC_readlen = N_GC.index.to_numpy(dtype=int)
    N_GC_gc = N_GC.columns.to_numpy(dtype=int)

    # sparse grid for measured read density
    F_GC_readlen = F_GC.index.to_numpy(dtype=int)
    F_GC_gc = F_GC.columns.to_numpy(dtype=int)

    N_f2 = csaps(
        [N_GC_readlen, N_GC_gc],
        N_GC.to_numpy(),
        smooth=smooth,
        normalizedsmooth=normalized,
    )
    F_f2 = csaps(
        [F_GC_readlen, F_GC_gc],
        F_GC.to_numpy(),
        smooth=smooth,
        normalizedsmooth=normalized,
    )

    scaling_dict = dict()
    for i in np.arange(N_GC_min, N_GC_max + 1, 1):
        readlen_tmp = i
        N_tmp = N_f2([readlen_tmp, N_GC_gc])
        F_tmp = F_f2([readlen_tmp, F_GC_gc])
        scaling_dict[i] = int(np.sum(N_tmp) / np.sum(F_tmp)) if np.sum(F_tmp) > 0 else 0

    # get dense data (full GC and readlen range)
    N_a, N_b = np.meshgrid(
        np.arange(N_GC_min, N_GC_max + 1, 1), N_GC.columns.to_numpy(dtype=int)
    )
    F_a, F_b = np.meshgrid(
        np.arange(F_GC_min, N_GC_max + 1, 1), F_GC.columns.to_numpy(dtype=int)
    )
    # convert to 2D coordinate pairs
    N_dense_points = np.stack([N_a.ravel(), N_b.ravel()], -1)

    r_list = list()
    f_list = list()
    n_list = list()
    for i in N_dense_points:
        x = i.tolist()
        scaling = scaling_dict[x[0]]
        if (N_f2(x)).astype(int) > minreads and (F_f2(x)).astype(int) > minreads:
            ratio = int(F_f2(x).item()) / int(N_f2(x).item()) * scaling
        else:
            ratio = 1
        f_int = int(F_f2(x).item())
        n_int = int(N_f2(x).item())
        f_list.append(f_int if f_int > 0 else 0)
        n_list.append(n_int if n_int > 0 else 0)
        r_list.append(ratio if ratio > 0 else 1)

    ratio_dense = np.array(r_list).reshape(N_a.shape).T
    F_dense = np.array(f_list).reshape(N_a.shape).T
    N_dense = np.array(n_list).reshape(N_a.shape).T

    # create indices for distributions
    ind_N = pd.MultiIndex.from_product([["N_gc"], np.arange(N_GC_min, N_GC_max + 1, 1)])
    ind_F = pd.MultiIndex.from_product([["F_gc"], np.arange(N_GC_min, N_GC_max + 1, 1)])
    ind_R = pd.MultiIndex.from_product([["R_gc"], np.arange(N_GC_min, N_GC_max + 1, 1)])
    # numpy to dataframe with indices
    NInt_df = pd.DataFrame(N_dense, columns=N_GC.columns, index=ind_N)
    FInt_df = pd.DataFrame(F_dense, columns=N_GC.columns, index=ind_F)
    RInt_df = pd.DataFrame(ratio_dense, columns=N_GC.columns, index=ind_R)

    return pd.concat(
        [NInt_df, FInt_df, RInt_df]
    )  # NInt_df.append(FInt_df).append(RInt_df)


def get_ratio(df, minreads=4):
    # separate hypothetical read density from measured read density
    N_GC = df.loc["N_gc"]
    F_GC = df.loc["F_gc"]
    # get min and max values
    # N_GC_min, N_GC_max = np.nanmin(N_GC.index.astype("int")), np.nanmax(N_GC.index.astype("int"))  # not used
    # F_GC_min, F_GC_max = np.nanmin(F_GC.index.astype("int")), np.nanmax(F_GC.index.astype("int"))  # not used

    scaling_dict = dict()
    for i in N_GC.index:
        n_tmp = N_GC.loc[i].to_numpy()
        f_tmp = F_GC.loc[i].to_numpy()
        scaling_dict[i] = int(np.sum(n_tmp) / np.sum(f_tmp)) if np.sum(f_tmp) > 0 else 0

    r_dict = dict()
    for i in N_GC.index:
        scaling = scaling_dict[i]
        f_gc_t = F_GC.loc[i]
        n_gc_t = N_GC.loc[i]
        r_gc_t = np.array(
            [
                float(f_gc_t[x]) / n_gc_t[x] * scaling
                if n_gc_t[x] and f_gc_t[x] > minreads
                else 1
                for x in range(len(f_gc_t))
            ]
        )
        r_dict[i] = r_gc_t

    ratio_dense = pd.DataFrame.from_dict(r_dict, orient="index", columns=N_GC.columns)
    ind = pd.MultiIndex.from_product([["R_gc"], ratio_dense.index])
    ratio_dense.index = ind

    return ratio_dense


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
    "--GCbiasFrequenciesFile",
    "-freq",
    "-o",
    "gcbias_frequency_output",
    required=True,
    type=click.Path(writable=True),
    help="""Path to save the file containing 
            the observed and expected read frequencies per %%GC-
            content. This file is needed to run the 
            correctGCBias tool. This is a tab separated file.""",
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
    "-ni",
    "--normalized_interpolation",
    "normalized_interpolation",
    is_flag=True,
    help="""Flag: option to normalize smooth parameters for interpolation. 
            If active, interpolation results are invariant to xdata range and 
            less sensitive to nonuniformity of weights and xdata clumping.""",
)
@click.option(
    "--minreads",
    "minreads",
    default=4,
    show_default=True,
    type=click.INT,
    help="""Minimum reads to consider, when calculating GC bias ratio.
    Bins with values below this threshold get a default value of 1 assigned.""",
)
@click.option(
    "--precomputed_background",
    "-pb",
    "precomputed_Ngc",
    type=click.Path(exists=True, readable=True),
    default=None,
    help="""A tsv. file containing parameters and a precomputed background 
    distribution created by the computeGCbias_background script. 
    Some command line options are overwritten make background and read distribution
    comparable. Make sure that genome builds match!""",
)
@click.option(
    "--MeasurementOutput",
    "-MO",
    "measurement_output",
    type=click.Path(writable=True),
    help="""Writes measured values to an output file.
            This option is only active is Interpolation is activated.""",
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
    gcbias_frequency_output,
    num_cpus,
    minlen,
    maxlen,
    lengthStep,
    interpolate,
    normalized_interpolation,
    minreads,
    measurement_output,
    precomputed_Ngc,
    sampleSize,
    blacklistfile,
    region,
    standard_chroms,
    verbose,
    debug,
    seed,
    mp_type,
):

    ### initial setup

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

    ### determine chromosome names for downstream analysis

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

    ### get regions for processing by sampling or by loading precomputed configuration
    if precomputed_Ngc:
        logger.info("Using precomputed background data.")
        logger.debug("Reading precomputed background data.")
        Ndata, param_dict = read_precomputed_table(precomputed_Ngc)
        bl_hash = param_dict["blacklist_hash"]
        if blacklistfile:
            logger.debug("Validating blacklist hashes.")
            assert bl_hash == hash_file(
                blacklistfile
            ), "Blacklist does not match blacklist used in precomputing the background distribtuion. Please use matching blacklists!"
        else:
            logger.debug("Validating that no blacklist was specified in both cases.")
            assert bl_hash == blacklistfile, "Blacklist was specified in one of the scripts, but not in the other. Please use matching blacklists!"
        logger.debug("Checking parameters for compatibility with provided bam file.")
        regions_params = param_dict["get_regions_params"]
        bam_dict = chrom_dict.copy()
        chrom_dict = regions_params["genome"]
        if chrom_dict.keys() != bam_dict.keys():
            precomputed_chrom_mapping = map_chroms(chrom_dict.keys(), bam_dict.keys())
            chrom_dict = {
                precomputed_chrom_mapping[key]: tuple(value)
                for key, value in chrom_dict.items()
            }
            regions_params["genome"] = chrom_dict
        else:
            chrom_dict = {key: tuple(value) for key, value in chrom_dict.items()}
            regions_params["genome"] = chrom_dict
        logger.info(
            f"Overwriting the following options: genome, sampleSize, region and seed"
        )
        logger.debug(f"params: {regions_params}")
        seed = regions_params["seed"]
        rng = np.random.default_rng(seed=seed)
        random.seed(seed)
        sampleSize = regions_params["nregions"]
        regions = get_regions(bam=bam, blacklist=blacklistfile, **regions_params)
    else:
        Ndata = None
        regions = get_regions(
            genome=chrom_dict,
            bam=bam,
            nregions=sampleSize,
            windowsize=1000,
            blacklist=blacklistfile,
            region=region,
            seed=seed,
        )


    sampleSize_regions = int(max(1,sampleSize / 1000))
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
        Ndata=Ndata,
    )
    # change the way data is handled
    if interpolate:
        if measurement_output:
            logger.info("saving measured data")
            data.to_csv(measurement_output, sep="\t")
        r_data = interpolate_ratio_csaps(data, minreads=minreads, normalized=normalized_interpolation)
        r_data.to_csv(gcbias_frequency_output, sep="\t")
    else:
        if measurement_output:
            logger.info(
                "Option MeasurementOutput has no effect. Measured data is saved in GCbiasFrequencies file!"
            )
        r_data = get_ratio(data, minreads=minreads)
        out_data = data.append(r_data)
        out_data.to_csv(gcbias_frequency_output, sep="\t")


if __name__ == "__main__":
    main()
