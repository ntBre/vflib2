import json
import logging
from argparse import ArgumentParser

from openff.qcsubmit.results import (
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.utils import _CachedPortalClient, portal_client_manager
from openff.toolkit import ForceField
from vflib2.config import Config
from vflib2.datasets import select_parameters
from vflib2.forcebalance import generate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = ArgumentParser()
parser.add_argument("config")

# these could be added to the config at some point
OPT_SMIRKS = "opt-smirks.json"
TD_SMIRKS = "td-smirks.json"


def curate_data(ff, opt, td, ring_torsions):
    """Select the parameters in ``ff`` to train based on the datasets ``opt``
    and ``td.``

    ``ring_torsions`` should be the name of a file containing a list of
    parameter IDs, one per line, that correspond to in-ring torsions. Other
    in-ring torsions are usually filtered out of the training data.

    .. note::
       This function calls ``*ResultCollection.to_records`` internally, so you
       may want to wrap calls to it with a cached ``PortalClient`` via
       ``openff.qcsubmit.utils.portal_client_manager``.
    """
    opt_smirks = select_parameters(opt, ["Bonds", "Angles"], ff)
    with open(OPT_SMIRKS, "w") as f:
        json.dump(opt_smirks, f, indent=2)

    td_smirks = select_parameters(
        td,
        ["ProperTorsions", "ImproperTorsions"],
        ff,
        ring_torsions,
    )
    with open(TD_SMIRKS, "w") as f:
        json.dump(td_smirks, f, indent=2)


def main():
    args = parser.parse_args()
    conf = Config.from_yaml(args.config)

    assert len(conf.opt_datasets) == 1, "Only 1 opt dataset can be used"
    opt = OptimizationResultCollection.parse_file(conf.opt_datasets[0])

    logger.info(f"loaded {opt.n_results} opt records")

    assert len(conf.td_datasets) == 1, "Only 1 td dataset can be used"
    td = TorsionDriveResultCollection.parse_file(conf.td_datasets[0])

    logger.info(f"loaded {td.n_results} td records")

    ff = ForceField(conf.initial_ff)

    # at least for now, I'm not doing any processing on the datasets, so I can
    # move straight into curating them
    client = _CachedPortalClient(
        "https://api.qcarchive.molssi.org:443", ".cache"
    )

    with portal_client_manager(lambda _: client):
        curate_data(ff, opt, td, conf.ring_torsions)

    # TODO skipping this for now
    if conf.do_msm:
        raise NotImplementedError("TODO: handle MSM request")

    # now prepare ForceBalance inputs
    generate(
        tag="fb-fit",
        optimization_dataset=opt,
        torsion_dataset=td,
        forcefield=conf.initial_ff,
        valence_to_optimize=OPT_SMIRKS,
        torsions_to_optimize=TD_SMIRKS,
        output_directory="output",
        smarts_to_exclude=conf.smarts_to_exclude,
        smiles_to_exclude=conf.smiles_to_exclude,
        verbose=True,
        max_iterations=50,
        port=55387,
    )


if __name__ == "__main__":
    main()
