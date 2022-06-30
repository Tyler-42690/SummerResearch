import json

class Config(object):
    def __init__(self, config_file = 'config.json') -> None:
        # Initialization stuff from a configuration file.
        self.readConfig(config_file)

    def readConfig(self, file):
        with open(file, "r") as f:
            dict = json.load(f)
            # Set configured attributes in the application
        for key, value in dict.items():
            setattr(self, key.strip(), value.strip())