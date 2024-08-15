from dataclasses import dataclass

import yaml


@dataclass
class Config:
    initial_ff: str
    opt_datasets: list[str]
    td_datasets: list[str]
    ring_torsions: str
    do_msm: bool

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            d = yaml.load(f, Loader=yaml.Loader)
            return cls(**d)
