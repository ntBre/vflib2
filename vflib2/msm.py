import faulthandler
import json
import logging
import os
from collections import defaultdict

import click
import numpy as np
import tqdm
from openff.qcsubmit.results import (
    BasicResultCollection,
    OptimizationResultCollection,
)
from openff.qcsubmit.results.filters import LowestEnergyFilter
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from qcportal.models import ResultRecord
from qube import Ligand, ModSeminario, ModSemMaths

logging.getLogger("openff").setLevel(logging.ERROR)


# this doesn't make sense, but for some reason monkey patching this (to the
# same function found in qubekit) has helped to avoid segfaults in the numpy
# dot product here. I think more recent versions of numpy also avoid this issue
def force_constant_bond(bond, eigenvals, eigenvecs, coords):
    atom_a, atom_b = bond
    eigenvals_ab = eigenvals[atom_a, atom_b, :]
    eigenvecs_ab = eigenvecs[:, :, atom_a, atom_b]

    unit_vectors_ab = ModSemMaths.unit_vector_along_bond(coords, bond)

    return -0.5 * sum(
        eigenvals_ab[i] * abs(np.dot(unit_vectors_ab, eigenvecs_ab[:, i]))
        for i in range(3)
    )


ModSemMaths.force_constant_bond = force_constant_bond


def calculate_parameters(
    qc_record: ResultRecord,
    molecule: Molecule,
    forcefield: ForceField,
) -> dict[str, dict[str, list[unit.Quantity]]]:
    """
    Calculate the modified seminario parameters for the given input molecule
    and store them by OFF SMIRKS.
    """
    mod_sem = ModSeminario()

    # create the qube molecule, this should be in the same order as the off_mol
    qube_mol = Ligand.from_rdkit(molecule.to_rdkit(), name="offmol")
    qube_mol.hessian = qc_record.return_result
    # calculate the modified seminario parameters and store in the molecule
    qube_mol = mod_sem.run(qube_mol)
    # label the openff molecule
    labels = forcefield.label_molecules(molecule.to_topology())[0]
    # loop over all bonds and angles and collect the results in
    # nm/ kj/mol / radians(openMM units)
    all_parameters = {
        "bond_eq": defaultdict(list),
        "bond_k": defaultdict(list),
        "angle_eq": defaultdict(list),
        "angle_k": defaultdict(list),
    }

    for bond, parameter in labels["Bonds"].items():
        # bond is a tuple of the atom index the parameter is applied to
        qube_param = qube_mol.BondForce[bond]
        all_parameters["bond_eq"][parameter.smirks].append(qube_param.length)
        all_parameters["bond_k"][parameter.smirks].append(qube_param.k)

    for angle, parameter in labels["Angles"].items():
        qube_param = qube_mol.AngleForce[angle]
        all_parameters["angle_eq"][parameter.smirks].append(qube_param.angle)
        all_parameters["angle_k"][parameter.smirks].append(qube_param.k)

    return all_parameters


@click.command()
@click.option(
    "--initial-force-field",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the initial force field file (OFFXML).",
)
@click.option(
    "--output",
    "output_force_field",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the output force field file (OFFXML).",
)
@click.option(
    "--optimization-dataset",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the optimization dataset.",
)
@click.option(
    "--working-directory",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    required=False,
    help=(
        "The path to the working directory. "
        "Intermediate files are saved here if provided"
    ),
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Enable verbose logging.",
)
def main(
    initial_force_field: str,
    output_force_field: str,
    optimization_dataset: str,
    working_directory: str | None,
    verbose: bool = False,
):
    optimization_dataset = OptimizationResultCollection.parse_file(
        optimization_dataset
    )
    _main(
        initial_force_field,
        output_force_field,
        optimization_dataset,
        working_directory,
        verbose,
    )


def _main(
    initial_force_field: str,
    output_force_field: str,
    dataset: OptimizationResultCollection,
    working_directory: str | None,
    verbose: bool = False,
):
    # filter for lowest energy results
    print("filtering")
    filtered = dataset.filter(LowestEnergyFilter())

    # filter to only keep entries with hessians calculated
    print("converting to results")
    hessian_set = filtered.to_basic_result_collection(driver="hessian")

    if working_directory is not None:
        if not os.path.exists(working_directory):
            os.mkdir(working_directory)
        hessian_file = os.path.join(working_directory, "hessian_set.json")
        with open(hessian_file, "w") as f:
            f.write(hessian_set.json(indent=2))
        if verbose:
            print(f"Hessian set written to: {hessian_file}")

    if verbose:
        print(f"Found {hessian_set.n_results} hessian calculations")
        print(f"Found {hessian_set.n_molecules} hessian molecules")

    ff = ForceField(initial_force_field, allow_cosmetic_attributes=True)

    records_and_molecules = list(hessian_set.to_records())
    if verbose:
        records_and_molecules = tqdm.tqdm(
            records_and_molecules,
            desc="Calculating parameters",
        )

    all_parameters = {
        "bond_eq": defaultdict(list),
        "bond_k": defaultdict(list),
        "angle_eq": defaultdict(list),
        "angle_k": defaultdict(list),
    }
    errored_records_and_molecules = []
    for record, molecule in records_and_molecules:
        try:
            parameters = calculate_parameters(record, molecule, ff)
        except BaseException:
            errored_records_and_molecules.append((record, molecule))
            continue
        else:
            for key, values in parameters.items():
                for smirks, value in values.items():
                    all_parameters[key][smirks].extend(value)

    if working_directory is not None:
        seminario_file = os.path.join(
            working_directory, "seminario_parameters.json"
        )
        with open(seminario_file, "w") as file:
            json.dump(all_parameters, file, indent=2)

    if verbose:
        print(
            f"Found {len(errored_records_and_molecules)} errored calculations"
        )
    if working_directory is not None:
        if len(errored_records_and_molecules):
            key = list(dataset.entries.keys())[0]
            opt_records_by_id = {
                record.record_id: record for record in hessian_set.entries[key]
            }
            records, _ = zip(*errored_records_and_molecules)
            errored_records = [
                opt_records_by_id[record.id] for record in records
            ]
            errored_dataset = BasicResultCollection(
                entries={key: errored_records}
            )
            error_file = os.path.join(
                working_directory, "errored_dataset.json"
            )
            with open(error_file, "w") as f:
                f.write(errored_dataset.json(indent=2))
            if verbose:
                print(f"Errored dataset written to: {error_file}")

    # now we need to update the FF parameters
    kj_per_mol_per_nm2 = unit.kilojoule_per_mole / unit.nanometer**2
    bond_handler = ff.get_parameter_handler("Bonds")
    for smirks in all_parameters["bond_eq"]:
        bond = bond_handler.parameters[smirks]

        bond_length = (
            np.mean(all_parameters["bond_eq"][smirks]) * unit.nanometer
        )
        bond.length = bond_length.to(unit.angstrom)

        bond_k = np.mean(all_parameters["bond_k"][smirks]) * kj_per_mol_per_nm2
        bond.k = bond_k.to(unit.kilocalorie_per_mole / (unit.angstrom**2))

    kj_per_mol_per_rad2 = unit.kilojoule_per_mole / (unit.radian**2)
    angle_handler = ff.get_parameter_handler("Angles")
    for smirks in all_parameters["angle_eq"]:
        angle = angle_handler.parameters[smirks]

        angle_eq = np.mean(all_parameters["angle_eq"][smirks]) * unit.radian
        angle.angle = angle_eq.to(unit.degree)

        angle_k = (
            np.mean(all_parameters["angle_k"][smirks]) * kj_per_mol_per_rad2
        )
        angle.k = angle_k.to(unit.kilocalorie_per_mole / unit.radian**2)

    ff.to_file(output_force_field)


if __name__ == "__main__":
    with open("fault_handler.log", "w") as fobj:
        faulthandler.enable(fobj)
        main()
