from pytorch_lightning.utilities.cli import LightningCLI
from readmission_datamodule import ReadmissionDatamodule
from code_only_transformer import SupportSetOnlyModel

cli = LightningCLI(SupportSetOnlyModel, ReadmissionDatamodule, save_config_overwrite=True)
