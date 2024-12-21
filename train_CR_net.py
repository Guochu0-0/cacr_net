import os
import yaml
import sys
import torch
import argparse
import numpy as np
from config import Config

from data.dataLoader import SEN12MSCR, sen12mscrts
from ca_cr_net.model_CR_net import ModelCRNet
from generic_train import Generic_Train
from model_base import seed_torch

##===================================================##
##********** Configure training settings ************##
##===================================================##

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='noCA', help='the name of config files')
parser.add_argument('--seed', type=int, default=0)
opts = parser.parse_args()
config = Config(os.path.join('config', f'{opts.config_name}.yml'))

##===================================================##
##****************** choose gpu *********************##
##===================================================##
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU

##===================================================##
##*************** Create dataloader *****************##
##===================================================##
seed_torch(opts.seed)

dt_train = sen12mscrts(config.TRAIN_ROOT, 
                        split='train', 
                        min_cov=config.MIN_COV,
                        max_cov=config.MAX_COV,
                        n_input_samples=config.INPUT_T,
                        sampler=config.SAMPLE_TYPE, 
                        import_data_path=config.TRAIN_IMPORT_PATH,
                        import_csv_path=config.TRAIN_PAIRS_PATH)
dt_val = sen12mscrts(config.VAL_ROOT, 
                        split='val', 
                        min_cov=config.MIN_COV,
                        max_cov=config.MAX_COV,
                        n_input_samples=config.INPUT_T,
                        import_data_path=config.VAL_IMPORT_PATH,
                        import_csv_path=config.VAL_PAIRS_PATH)
dt_test = sen12mscrts(config.TEST_ROOT, 
                    split='test', 
                    min_cov=config.MIN_COV,
                    max_cov=config.MAX_COV,
                    n_input_samples=config.INPUT_T,
                    import_data_path=config.TEST_IMPORT_PATH,
                    import_csv_path=config.TEST_PAIRS_PATH)

if config.SUB_NUM:
    dt_train = torch.utils.data.Subset(dt_train, np.random.choice(len(dt_train), config.SUB_NUM, replace=False))
    dt_val = torch.utils.data.Subset(dt_val, np.random.choice(len(dt_train), config.SUB_NUM, replace=False))

train_loader = torch.utils.data.DataLoader(
    dt_train,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    dt_val,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    dt_test,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

print("Train {}, Val {}".format(len(dt_train), len(dt_val)))

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelCRNet(config)

##===================================================##
##**************** Train the network ****************##
##===================================================##
Generic_Train(model, config, train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader).train()
