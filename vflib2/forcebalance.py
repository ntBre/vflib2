"""Functions for generating ForceBalance input"""

import json
import logging
import os
import typing
from pathlib import Path

from openff.bespokefit.optimizers.forcebalance import ForceBalanceInputFactory
from openff.bespokefit.schema.fitting import (
    OptimizationSchema,
    OptimizationStageSchema,
)
from openff.bespokefit.schema.optimizers import ForceBalanceSchema
from openff.bespokefit.schema.smirnoff import (
    AngleHyperparameters,
    AngleSMIRKS,
    BondHyperparameters,
    BondSMIRKS,
    ImproperTorsionHyperparameters,
    ImproperTorsionSMIRKS,
    ProperTorsionHyperparameters,
    ProperTorsionSMIRKS,
)
from openff.bespokefit.schema.targets import (
    OptGeoTargetSchema,
    TorsionProfileTargetSchema,
)
from openff.qcsubmit.results import (
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.results.filters import SMARTSFilter, SMILESFilter
from openff.toolkit import ForceField

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_training_data(
    optimization_dataset: OptimizationResultCollection,
    torsion_dataset: TorsionDriveResultCollection,
    smarts_to_exclude: typing.Optional[str] = None,
    smiles_to_exclude: typing.Optional[str] = None,
    verbose: bool = False,
):
    if smarts_to_exclude is not None:
        exclude_smarts = Path(smarts_to_exclude).read_text().splitlines()
    else:
        exclude_smarts = []

    if smiles_to_exclude is not None:
        exclude_smiles = Path(smiles_to_exclude).read_text().splitlines()
    else:
        exclude_smiles = []

    torsion_training_set = torsion_dataset
    if verbose:
        n = torsion_training_set.n_results
        logger.info(f"Loaded torsion training set with {n} entries.")

    torsion_training_set = torsion_training_set.filter(
        SMARTSFilter(smarts_to_exclude=exclude_smarts),
        SMILESFilter(smiles_to_exclude=exclude_smiles),
    )

    if verbose:
        n = torsion_training_set.n_results
        logger.info(f"Filtered torsion training set to {n} entries.")

    optimization_training_set = optimization_dataset
    if verbose:
        n = optimization_training_set.n_results
        logger.info(f"Loaded optimization training set with {n} entries.")
    optimization_training_set = optimization_training_set.filter(
        SMARTSFilter(smarts_to_exclude=exclude_smarts),
        SMILESFilter(smiles_to_exclude=exclude_smiles),
    )
    if verbose:
        n = optimization_training_set.n_results
        logger.info(f"Filtered optimization training set to {n} entries.")

    return torsion_training_set, optimization_training_set


def generate(
    tag: str,
    optimization_dataset: OptimizationResultCollection,
    torsion_dataset: TorsionDriveResultCollection,
    forcefield: str,
    valence_to_optimize: str,
    torsions_to_optimize: str,
    output_directory: str,
    smarts_to_exclude: typing.Optional[str] = None,
    smiles_to_exclude: typing.Optional[str] = None,
    verbose: bool = False,
    max_iterations: int = 50,
    port: int = 55387,
):

    torsion_training_set, optimization_training_set = load_training_data(
        optimization_dataset=optimization_dataset,
        torsion_dataset=torsion_dataset,
        smarts_to_exclude=smarts_to_exclude,
        smiles_to_exclude=smiles_to_exclude,
        verbose=verbose,
    )

    optimizer = ForceBalanceSchema(
        max_iterations=max_iterations,
        step_convergence_threshold=0.01,
        objective_convergence_threshold=0.1,
        gradient_convergence_threshold=0.1,
        n_criteria=2,
        initial_trust_radius=-1.0,
        finite_difference_h=0.01,
        extras={
            "wq_port": str(port),
            "asynchronous": "True",
            "search_tolerance": "0.1",
            "backup": "0",
            "retain_micro_outputs": "0",
        },
    )

    targets = [
        TorsionProfileTargetSchema(
            reference_data=torsion_training_set,
            energy_denominator=1.0,
            energy_cutoff=8.0,
            extras={"remote": "1"},
        ),
        OptGeoTargetSchema(
            reference_data=optimization_training_set,
            weight=0.01,
            extras={"batch_size": 30, "remote": "1"},
            bond_denominator=0.05,
            angle_denominator=5.0,
            dihedral_denominator=10.0,
            improper_denominator=10.0,
        ),
    ]

    # a16, a17, a27, a35
    linear_angle_smirks = [
        "[*:1]~[#6X2:2]~[*:3]",  # a16
        "[*:1]~[#7X2:2]~[*:3]",  # a17
        "[*:1]~[#7X2:2]~[#7X1:3]",  # a27
        "[*:1]=[#16X2:2]=[*:3]",
    ]  # a35, this one anyways doesn't have a training target for ages

    with open(valence_to_optimize, "r") as f:
        valence_smirks = json.load(f)
    with open(torsions_to_optimize, "r") as f:
        torsion_smirks = json.load(f)

    target_parameters = []
    for smirks in valence_smirks["Angles"]:
        if smirks in linear_angle_smirks:
            parameter = AngleSMIRKS(smirks=smirks, attributes={"k"})
        else:
            parameter = AngleSMIRKS(smirks=smirks, attributes={"k", "angle"})
        target_parameters.append(parameter)

    for smirks in valence_smirks["Bonds"]:
        target_parameters.append(
            BondSMIRKS(smirks=smirks, attributes={"k", "length"})
        )

    ff = ForceField(forcefield, allow_cosmetic_attributes=True)
    try:
        ff.deregister_parameter_handler("Constraints")
    except KeyError:
        logger.warn(
            "failed to deregister constraint handler, already unconstrained?"
        )

    torsion_handler = ff.get_parameter_handler("ProperTorsions")
    for smirks in torsion_smirks["ProperTorsions"]:
        if smirks in torsion_handler.parameters:
            original_k = torsion_handler.parameters[smirks].k
            attributes = {f"k{i + 1}" for i in range(len(original_k))}
            target_parameters.append(
                ProperTorsionSMIRKS(smirks=smirks, attributes=attributes)
            )

    improper_handler = ff.get_parameter_handler("ImproperTorsions")
    for smirks in torsion_smirks["ImproperTorsions"]:
        if smirks in improper_handler.parameters:
            original_k = improper_handler.parameters[smirks].k
            attributes = {f"k{i + 1}" for i in range(len(original_k))}
            target_parameters.append(
                ImproperTorsionSMIRKS(smirks=smirks, attributes=attributes)
            )

    # use the absolute path for an existing offxml file or assume builtin and
    # use the exact input
    if os.path.exists(forcefield):
        ff_path = os.path.abspath(forcefield)
    else:
        ff_path = forcefield

    optimization_schema = OptimizationSchema(
        id=tag,
        initial_force_field=ff_path,
        stages=[
            OptimizationStageSchema(
                optimizer=optimizer,
                targets=targets,
                parameters=target_parameters,
                parameter_hyperparameters=[
                    AngleHyperparameters(priors={"k": 100, "angle": 5}),
                    BondHyperparameters(priors={"k": 100, "length": 0.1}),
                    ProperTorsionHyperparameters(priors={"k": 5}),
                    ImproperTorsionHyperparameters(priors={"k": 5}),
                ],
            )
        ],
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    optdir = output_directory / "schemas" / "optimizations"
    optdir.mkdir(parents=True, exist_ok=True)

    optfile = optdir / f"{optimization_schema.id}.json"
    with optfile.open("w") as f:
        f.write(optimization_schema.json(indent=2))

    ff = ForceField(
        optimization_schema.initial_force_field,
        allow_cosmetic_attributes=True,
    )
    try:
        ff.deregister_parameter_handler("Constraints")
    except KeyError:
        logger.warn(
            "failed to deregister constraint handler, already unconstrained?"
        )
    # Generate the ForceBalance inputs
    ForceBalanceInputFactory.generate(
        os.path.join(optimization_schema.id),
        optimization_schema.stages[0],
        ff,
    )


if __name__ == "__main__":
    generate()
