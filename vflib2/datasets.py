import functools
import typing
from collections import Counter, defaultdict

import numpy as np
from openff.qcsubmit.results import (
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.toolkit import ForceField, Molecule
from openff.toolkit.utils.exceptions import NotBondedError
from qcportal.torsiondrive.record_models import (
    OptimizationRecord,
    TorsiondriveRecord,
)

Record = typing.Union[TorsiondriveRecord, OptimizationRecord]


def safe_ring_bond(mol: Molecule, i: int, j: int) -> bool:
    """Returns true if the bond between atoms ``i`` and ``j`` in ``mol`` is in
    a ring. Catches ``NotBondedError``s and returns ``False`` instead.

    This function will always return ``False`` for improper torsions, but I
    think an improper with three ring bonds will be fairly rare anyway.

    """
    try:
        return mol.get_bond_between(i, j).is_in_ring()
    except NotBondedError:
        return False


def check_torsion_is_in_ring(
    mol: Molecule,
    indices: typing.Tuple[int, int, int, int],
) -> bool:
    """
    Check if a torsion is in a ring.

    If a torsion I-J-K-L is given, it checks
    whether all bonds I-J, J-K, and K-L are in a ring.

    """
    i, j, k, m = indices
    return (
        safe_ring_bond(mol, i, j)
        and safe_ring_bond(mol, j, k)
        and safe_ring_bond(mol, k, m)
    )


def label_and_tag_ids(
    record_and_molecule: typing.Tuple[Record, Molecule],
    force_field: ForceField,
    parameter_types: typing.List[str],
    explicit_ring_torsions: typing.Optional[str] = None,
) -> typing.Set[typing.Tuple[str, str, int]]:
    if explicit_ring_torsions is not None:
        ring_torsions = np.loadtxt(explicit_ring_torsions, dtype=str)
    else:
        ring_torsions = []

    record, molecule = record_and_molecule
    mol_labels = force_field.label_molecules(molecule.to_topology())[0]
    parameter_ids = set()

    for parameter_type in parameter_types:
        parameter_labels = mol_labels[parameter_type]

        for indices, parameter in parameter_labels.items():
            # remove mismatching torsiondrives
            if isinstance(record, TorsiondriveRecord):
                # check central bond, i.e. middle 2 atoms
                record_atoms = record.specification.keywords.dihedrals[0]
                if set(indices[1:3]) != set(record_atoms[1:3]):
                    continue

                # some general parameters overlap with in-ring torsions and
                # there are many torsion scans from Gen1 sets that have
                # in-ring torsions and we want to exclude them in training
                # as they result in higher k values unless the parameters
                # have smirks explicitly for an in-ring torsion. It is to be
                # noted that training on in-ring torsions is needed to
                # properly model puckering in rings with hetero atoms
                if parameter.id not in ring_torsions:
                    if check_torsion_is_in_ring(molecule, indices):
                        continue

            n_heavy_atoms = sum(
                1 for atom in molecule.atoms if atom.atomic_number != 1
            )
            parameter_ids.add((parameter.id, record.id, n_heavy_atoms))
    return parameter_ids


def get_parameter_distribution(
    dataset: typing.Union[
        TorsionDriveResultCollection, OptimizationResultCollection
    ],
    parameter_types: typing.List[str],
    force_field: ForceField,
    explicit_ring_torsions: typing.Optional[str] = None,
    n_processes: int = 4,
) -> typing.Tuple[
    Counter, typing.Dict[str, typing.List[typing.Tuple[int, str]]]
]:
    coverage = Counter()
    parameter_records = defaultdict(list)

    func = functools.partial(
        label_and_tag_ids,
        force_field=force_field,
        parameter_types=parameter_types,
        explicit_ring_torsions=explicit_ring_torsions,
    )
    for record in dataset.to_records():
        parameter_ids = func(record)
        for parameter_id, record_id, n_heavy_atoms in parameter_ids:
            coverage[parameter_id] += 1
            parameter_records[parameter_id].append((n_heavy_atoms, record_id))

    return coverage, dict(parameter_records)


def select_parameters(
    dataset: typing.Union[
        TorsionDriveResultCollection, OptimizationResultCollection
    ],
    parameter_types: typing.List[str],
    force_field: ForceField,
    explicit_ring_torsions: typing.Optional[str] = None,
    n_processes: int = 1,
    min_coverage: int = 5,
):
    coverage, _ = get_parameter_distribution(
        dataset=dataset,
        parameter_types=parameter_types,
        force_field=force_field,
        explicit_ring_torsions=explicit_ring_torsions,
        n_processes=n_processes,
    )

    selected_parameters = defaultdict(list)
    for parameter_type in parameter_types:
        handler = force_field.get_parameter_handler(parameter_type)

        for parameter_id, count in coverage.items():
            if count < min_coverage:
                continue
            parameters = handler.get_parameter({"id": parameter_id})
            if not len(parameters):
                continue
            selected_parameters[parameter_type].append(parameters[0].smirks)
    return selected_parameters
