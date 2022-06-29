# cfDNA GCcorrection

## User-friendly tools for correction GC biases in cell-free DNA samples

cfDNA_GCcorrection is an easy-to-use tool to determine and correct GC biases in cell-free DNA (cfDNA) samples. Based on fragment length and GC distributions, it calculates read-level correction values and makes them available as tags in SAM/BAM format. This additional information can be used during signal extraction in various metrics (e.g., allelic balance, coverage, read ends/midpoints, WPS), while preserving the original read coverage patterns for specific analyses.


### Documentation:

Under construction

### Installation

**Install by cloning this repository:**

You can install cfDNA_GCcorrection on command line (linux/mac) by cloning this git repository :

``` bash
$ git clone https://github.com/kircherlab/cfDNA_GCcorrection.git
$ cd cfDNA_GCcorrection
$ python setup.py install
```

By default, the script will install the python library and executable
codes globally, which means you need to be root or administrator of
the machine to complete the installation. If you need to
provide a nonstandard install prefix, or any other nonstandard
options, you can provide many command line options to the install
script.

	$ python setup.py --help

For example, to install under a specific location use:

	$ python setup.py install --prefix <target directory>

To install into your home directory, use:

	$ python setup.py install --user


## Acknowledgment

cfDNA_GCcorrection started as a fork of [deepTools](https://github.com/deeptools/deepTools) and was used as a skeleton for developing a GCcorrection method tailored towards cell-free DNA samples.
