from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from classification_model import ClassificationModel
from classification_dataset import MIMICClassificationDataModule

cli = LightningCLI(ClassificationModel, MIMICClassificationDataModule, save_config_overwrite=True,
                   trainer_defaults={'plugins': DDPPlugin(find_unused_parameters=False)})
