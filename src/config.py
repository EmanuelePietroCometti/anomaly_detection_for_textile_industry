import os
import yaml

def load_config(config_path=None):
    """
    Resolution order:
      1. explicit `config_path` argument
      2. the AD_CONFIG environment variable (used by the SLURM sweep)
      3. the repository default, "config.yaml"
    """
    if config_path is None:
        config_path = os.environ.get("AD_CONFIG", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)