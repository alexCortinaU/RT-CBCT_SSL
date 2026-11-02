from omegaconf import OmegaConf

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.utils import extras, get_metric_value, task_wrapper

# `${zfill:${pretrained_epoch},3}` -> "099", used to build pretrained checkpoint paths in the configs
if not OmegaConf.has_resolver("zfill"):
    OmegaConf.register_new_resolver("zfill", lambda value, width: str(value).zfill(int(width)))
