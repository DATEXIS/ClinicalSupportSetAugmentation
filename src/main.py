from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from model import SupportSetModel
from dataset import MIMICClassificationDataModule

cli = LightningCLI(SupportSetModel, MIMICClassificationDataModule, save_config_overwrite=True)
