from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from classification_model import ClassificationModel
from readmission_datamodule import ReadmissionDatamodule

cli = LightningCLI(ClassificationModel, ReadmissionDatamodule, save_config_overwrite=True)
