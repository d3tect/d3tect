import yaml

def _read_yaml_file(filename):
    with open(filename, 'r') as stream:
            return yaml.safe_load(stream)