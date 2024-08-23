# vflib2
tools for optimizing valence terms in OpenFF force fields

## Installation

Clone the repository and install into your current (conda) environment:

``` shell
pip install -e .
```

## Usage

The primary usage is through the included `vfgenerate.py` script, which should
be available on your `PATH` after installation.

``` shell
python vfgenerate.py config.yaml
```

This will follow the general steps from our `valence-fitting`
repositories[[1]][1][[2]][2], and from the [Sage 2.1.0][sage-2.1] and [Sage
2.2.0][sage-2.2] repositories:

1. Loading and curating an `OptimizationResultCollection` and a
   `TorsionDriveResultCollection` in the [qcsubmit][qcsubmit] JSON format.
2. Generating an initial modified Seminario method (MSM) guess for the bond and
   angle parameters.
3. Generating [ForceBalance][fb] input ready to run on a compute cluster.

### Input file

The input should be in YAML format and contains the following options:

| Option              | Type        | Description                                                         |
|---------------------|-------------|---------------------------------------------------------------------|
| `initial_ff`        | `str`       | Path to an OFFXML file (or built-in force field)                    |
| `opt_datasets`      | `list[str]` | Paths to [qcsubmit][qcsubmit] optimization datasets in JSON format  |
| `td_datasets`       | `list[str]` | Paths to [qcsubmit][qcsubmit] torsion drive datasets in JSON format |
| `ring_torsions`     | `str`       | Path to a file containing explicit ring torsions                    |
| `do_msm`            | `bool`      | Whether or not to replace bonds and angles with an MSM guess        |
| `smarts_to_exclude` | `str`       | SMARTS patterns to exclude from [ForceBalance][fb] training         |
| `smiles_to_exclude` | `str`       | SMILES strings to exclude from [ForceBalance][fb] training          |

The `opt_datasets` and `td_datasets` options currently allow providing multiple
datasets, but only single datasets of each kind are currently supported. You
should make sure your training datasets are pre-filtered and combined into
single JSON files before passing to `vfgenerate.py`.

#### Example

Here's an example input file from my first test run of `vfgenerate.py`:

``` yaml
initial_ff: tm_besmarts_reset_on_gen2.offxml

opt_datasets:
    - ../../valence-fitting/02_curate-data/datasets/combined-opt.json

td_datasets:
    - ../../valence-fitting/02_curate-data/datasets/combined-td.json

ring_torsions: ../../valence-fitting/02_curate-data/explicit_ring_torsions.dat

do_msm: false

smarts_to_exclude: ../../valence-fitting/04_fit-forcefield/smarts-to-exclude.dat
smiles_to_exclude: ../../valence-fitting/04_fit-forcefield/smiles-to-exclude.dat
```

`explicit_ring_torsions.dat` contains a sequence of OpenFF parameter IDs, one
per line:

``` shell
$ head ../../valence-fitting/02_curate-data/explicit_ring_torsions.dat
t15
t44
t49
t80
t84
```

Similarly, the SMARTS and SMILES files contain one pattern per line:

``` shell
$ head ../../valence-fitting/04_fit-forcefield/smarts-to-exclude.dat ../../valence-fitting/04_fit-forcefield/smiles-to-exclude.dat
==> ../../valence-fitting/04_fit-forcefield/smarts-to-exclude.dat <==
[#8+1:1]=[#7:2]
[#15:1]=[#6:2]
[#16+1:1]~[*:2]
[*:1]=[#15:2]-[#7:3]~[*:4]
[#17:1]~[#1:2]
[#16-1:1]~[#15:2]
[#7:1]=[#7:2]#[#7:3]
[#16-1:1]~[#16:2]
==> ../../valence-fitting/04_fit-forcefield/smiles-to-exclude.dat <==
[S](=[N-])(=O)
[H]C([H])(C#N)N=N#N
O[O-]
[H]c1c(c(c(c(c1[H])[H])[N@](C([H])([H])[H])[S@](=O)N(C([H])([H])[H])C([H])([H])[H])[H])[H]
```

These are obviously optional conceptually, but the code currently expects the
files to exist. However, it should handle empty files gracefully.

<!-- References -->
[1]: https://github.com/lilyminium/valence-fitting
[2]: https://github.com/ntBre/valence-fitting/
[sage-2.1]: https://github.com/openforcefield/sage-2.1.0
[sage-2.2]: https://github.com/openforcefield/sage-2.2.0
[qcsubmit]: https://github.com/openforcefield/openff-qcsubmit
[fb]: https://github.com/leeping/forcebalance
