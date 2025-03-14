from ruamel.yaml import YAML


class YamlParser:
    def __init__(self, path):
        stream = open(path, "r")
        yaml = YAML()
        yaml_args = yaml.load_all(stream)
        self._config = {}

        for data in yaml_args:
            self._config = dict(data)

    def get_config(self):
        return self._config
