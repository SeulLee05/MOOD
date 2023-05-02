import argparse
from parsers.parser import Parser
from parsers.config import get_config

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def main(work_type_args):
    args = Parser().parse()
    config = get_config(args.config, args.gpu, args.seed)

    if work_type_args.type == 'train':
        from prop_trainer import Trainer
        Trainer(config).train()

    elif work_type_args.type == 'sample':
        from sampler import Sampler
        Sampler(config).sample()

    else:
        raise ValueError(f'Wrong type {work_type_args.type}')


if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('-t', '--type', type=str, required=True)

    main(work_type_parser.parse_known_args()[0])
