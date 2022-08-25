# cfDNA GCcorrection

## User-friendly tools for correction GC biases in cell-free DNA samples

cfDNA_GCcorrection is an easy-to-use tool to determine and correct GC biases in cell-free DNA (cfDNA) samples. Based on fragment length and GC distributions, it calculates read-level correction values and makes them available as tags in SAM/BAM format. This additional information can be used during signal extraction in various metrics (e.g., allelic balance, coverage, read ends/midpoints, WPS), while preserving the original read coverage patterns for specific analyses.


### Documentation:

Under construction

### Installation

**Install by cloning this repository:**

You can install cfDNA_GCcorrection on command line (linux/mac) by cloning this git repository :

``` bash
git clone https://github.com/kircherlab/cfDNA_GCcorrection.git
cd cfDNA_GCcorrection
pip install -e .
```

**note:** pybedtools needs a locally installed version of bedtools! Please install it by other means. See the official [documentation](https://bedtools.readthedocs.io/en/latest/content/installation.html).

## Acknowledgment

cfDNA_GCcorrection started as a fork of [deepTools](https://github.com/deeptools/deepTools) and was used as a skeleton for developing a GCcorrection method tailored towards cell-free DNA samples.
