import argparse


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-t', '--type', type=str, required=True)

        self.set_arguments()

    def set_arguments(self):
        self.parser.add_argument('-c', '--config', type=str, required=True, help="Path of config file")
        self.parser.add_argument('-g', '--gpu', type=str, default='0', help="gpu")
        self.parser.add_argument('-s', '--seed', type=int, default=42)
        
    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit(f'Unknown argument: {unparsed}')
        return args
