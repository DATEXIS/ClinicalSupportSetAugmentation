from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from model import SupportSetModel
from readmission_datamodule import ReadmissionDatamodule

cli = LightningCLI(SupportSetModel, ReadmissionDatamodule, save_config_overwrite=True)
