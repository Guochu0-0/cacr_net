import os
import sys
import csv
import random
import time
#print(sys.path)
sys.path.append("/remote-home/chuguoyou/Code/CR/CR")
import glob
import torch
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from datetime import datetime
to_date   = lambda string: datetime.strptime(string, '%Y-%m-%d')
S1_LAUNCH = to_date('2014-04-03')

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
from s2cloudless import S2PixelCloudDetector

import rasterio
from rasterio.merge import merge
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
#from stocaching import SharedCache
from util.detect_cloudshadow import get_cloud_mask, get_shadow_mask


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif

def read_img(tif):
    return tif.read().astype(np.float32)

def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img      = (img - oldMin) / oldRange
    return img

def process_MS(img, method):
    if method=='default':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img = rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
        #img = img * 2 - 1

    if method=='resnet':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img /= 2000                                        # project to [0,5], preserve global intensities (across patches)
    img = np.nan_to_num(img)
    return torch.tensor(img)

def process_SAR(img, method):
    if method=='default':
        dB_min, dB_max = -25, 0                            # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                 # intensity clipping to a global unified SAR dB range
        img = rescale(img, dB_min, dB_max)                 # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
        #img = img * 2 - 1

    if method=='resnet':
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate([(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                              (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    img = np.nan_to_num(img)
    return torch.tensor(img)

def get_cloud_cloudshadow_mask(img, cloud_threshold=0.2):
    cloud_mask = get_cloud_mask(img, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(img)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0] = 1

    # label clouds and shadows
    cloud_cloudshadow_mask[cloud_cloudshadow_mask != 0] = 1
    return cloud_cloudshadow_mask


# recursively apply function to nested dictionary
def iterdict(dictionary, fct):
    for k,v in dictionary.items():        
        if isinstance(v, dict):
            dictionary[k] = iterdict(v, fct)
        else:      
            dictionary[k] = fct(v)      
    return dictionary

def get_cloud_map(img, detector, instance=None):

    # get cloud masks
    img = np.clip(img, 0, 10000)
    mask = np.ones((img.shape[-1], img.shape[-1]))
    # note: if your model may suffer from dark pixel artifacts,
    #       you may consider adjusting these filtering parameters
    if not (img.mean()<1e-5 and img.std() <1e-5):
        if detector == 'cloud_cloudshadow_mask':
            threshold = 0.2  # set to e.g. 0.2 or 0.4
            mask = get_cloud_cloudshadow_mask(img, threshold)
        elif detector== 's2cloudless_map':
            threshold = 0.5
            mask = instance.get_cloud_probability_maps(np.moveaxis(img/10000, 0, -1)[None, ...])[0, ...]
            mask[mask < threshold] = 0
            mask = gaussian_filter(mask, sigma=2)
        elif detector == 's2cloudless_mask':
            mask = instance.get_cloud_masks(np.moveaxis(img/10000, 0, -1)[None, ...])[0, ...]
        else:
            mask = np.ones((img.shape[-1], img.shape[-1]))
            warnings.warn(f'Method {detector} not yet implemented!')
    else:   warnings.warn(f'Encountered a blank sample, defaulting to cloudy mask.')
    return mask.astype(np.float32)


# function to fetch paired data, which may differ in modalities or dates
def get_pairedS1(patch_list, root_dir, mod=None, time=None):
    paired_list = []
    for patch in patch_list:
        seed, roi, modality, time_number, fname = patch.split('/')
        time = time_number if time is None else time # unless overwriting, ...
        mod  = modality if mod is None else mod      # keep the patch list's original time and modality
        n_patch       = fname.split('patch_')[-1].split('.tif')[0]
        paired_dir    = os.path.join(seed, roi, mod.upper(), str(time))
        candidates    = os.path.join(root_dir, paired_dir, f'{mod}_{seed}_{roi}_ImgNo_{time}_*_patch_{n_patch}.tif')
        paired_list.append(os.path.join(paired_dir, os.path.basename(glob.glob(candidates)[0])))
    return paired_list


    
################################################################################################################################################################################
# a clean sen12mscrts dataset
# only support import from pre-computed cloud coverages 

class sen12mscrts(Dataset):
    def __init__(self, root, split="all", region='all', cloud_masks=None, 
                 sample_type='cloudy_cloudfree', n_input_samples=3, 
                 rescale_method='default', min_cov=0.0, max_cov=1.0, 
                 import_data_path=None, import_csv_path=None, dist=1):
        
        self.root_dir = root   # set root directory which contains all ROI
        self.region   = region # region according to which the ROI are selected
        self.split = split

        self.modalities     = ["S1", "S2"]
        self.time_points    = range(30)
        self.cloud_masks    = cloud_masks  # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.sample_type    = sample_type #if self.cloud_masks is not None else 'generic' # pick 'generic' or 'cloudy_cloudfree'
        self.n_input_t      = n_input_samples  # specifies the number of samples, if only part of the time series is used as an input
        self.dist           = dist

        if self.cloud_masks in ['s2cloudless_map', 's2cloudless_mask']:
            self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        else: self.cloud_detector = None

        self.import_data_path = import_data_path
        self.statistics = np.load(import_data_path, allow_pickle=True).item()
        self.n_statistics = len(self.statistics)

        self.paths            = torch.load(f"data/precomputed/sen12mscrts/paths/{split}_{region}_path.pth")

        self.n_samples        = len(self.paths)
        # raise a warning that no data has been found
        if not self.n_samples: self.throw_warn()

        self.method          = rescale_method
        self.min_cov, self.max_cov = min_cov, max_cov

        if import_csv_path is not None:
            self.pairs = []
            with open(import_csv_path, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # 跳过标题行
                for row in reader:
                    cloudy = int(row[1])
                    cloudless = int(row[2])
                    self.pairs.append([cloudy, cloudless])
            
            assert len(self.pairs)==self.n_samples , "length of precomputed cloudy-cloudless pairs must match current dataset"
            assert len(self.pairs)==self.n_statistics , "length of precomputed covorages must match current dataset"

    def throw_warn(self):
        warnings.warn("""No data samples found! Please use the following directory structure:
                        
        path/to/your/SEN12MSCRTS/directory:
        ├───ROIs1158
        ├───ROIs1868
        ├───ROIs1970
        │   ├───20
        │   ├───21
        │   │   ├───S1
        │   │   └───S2
        │   │       ├───0
        │   │       ├───1
        │   │       │   └─── ... *.tif files
        │   │       └───30
        │   ...
        └───ROIs2017
                        
        Note: the data is provided by ROI geo-spatially separated and sensor modalities individually.
        You can simply merge the downloaded & extracted archives' subdirectories via 'mv */* .' in the parent directory
        to obtain the required structure specified above, which the data loader expects.
        """)

    def rf_sampler(self, idx, coverage, clear_tresh = 1e-3, seed=None):
        if self.split != 'train':
            seed = 4068

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        cloudy_idx, cloudless_idx = self.pairs[idx]
        min_cov = max(self.min_cov, clear_tresh)

        specific_idx = [
            pdx for pdx, perc in enumerate(coverage) 
            if min_cov <= perc <= self.max_cov
        ]

        # soft the coverage constraint
        if len(specific_idx) < self.n_input_t+1:
            specific_idx = [
                pdx for pdx, perc in enumerate(coverage) 
                if perc >= clear_tresh
            ]

        if len(specific_idx) < self.n_input_t+1:
            specific_idx = [pdx for pdx in range(30)]

        inputs_idx = []
        inputs_idx.append(cloudy_idx)
        remaining_indices = [i for i in specific_idx if i != cloudless_idx and i != cloudy_idx]
        reference_idx = random.sample(remaining_indices, self.n_input_t-1)

        inputs_idx += reference_idx

        return inputs_idx, cloudless_idx


    # load images at a given patch pdx for given time points tdx
    def get_imgs(self, pdx, tdx=range(0,30)):
        # load the images and infer the masks
        s1_tif   = [read_tif(os.path.join(self.root_dir, img)) for img in np.array(self.paths[pdx]['S1'])[tdx]]
        s2_tif   = [read_tif(os.path.join(self.root_dir, img)) for img in np.array(self.paths[pdx]['S2'])[tdx]]
        s1       = [process_SAR(read_img(img), self.method) for img in s1_tif]
        s2       = [read_img(img) for img in s2_tif]  # note: pre-processing happens after cloud detection
        masks    = None if not self.cloud_masks else [get_cloud_map(img, self.cloud_masks, self.cloud_detector) for img in s2]
        s2       = [process_MS(img, self.method) for img in s2]

        # get statistics and additional meta information
        s1_dates = [to_date(img.split('/')[-1].split('_')[5]) for img in np.array(self.paths[pdx]['S1'])[tdx]]
        s2_dates = [to_date(img.split('/')[-1].split('_')[5]) for img in np.array(self.paths[pdx]['S2'])[tdx]]
        s1_td    = [(date-S1_LAUNCH).days for date in s1_dates]
        s2_td    = [(date-S1_LAUNCH).days for date in s2_dates]

        return s1, s2, s1_td, s2_td, masks

    def getsample(self, pdx):
        return self.__getitem__(pdx)

    def __getitem__(self, pdx):  # get the time series of one patch
        coverage = [stats.item() for stats in self.statistics[pdx]['coverage']]

        inputs_idx, cloudless_idx = self.rf_sampler(pdx, coverage)

        in_s1, in_s2, in_s1_td, in_s2_td, in_masks = self.get_imgs(pdx, inputs_idx)
        tg_s1, tg_s2, tg_s1_td, tg_s2_td, tg_masks = self.get_imgs(pdx, [cloudless_idx])


        sample = {'input': {'S1': in_s1,
                            'S2': in_s2,
                            #'masks': list(in_masks),
                            'S1 TD': in_s1_td, #[s1_td[idx] for idx in inputs_idx],
                            'S2 TD': in_s2_td, #[s2_td[idx] for idx in inputs_idx],
                            'S1 path': [os.path.join(self.root_dir, self.paths[pdx]['S1'][idx]) for idx in inputs_idx],
                            'S2 path': [os.path.join(self.root_dir, self.paths[pdx]['S2'][idx]) for idx in inputs_idx],
                            },
                'target': {'S1': tg_s1,
                            'S2': tg_s2,
                            #'masks': [tg_masks],
                            'S1 path': [os.path.join(self.root_dir, self.paths[pdx]['S1'][cloudless_idx])],
                            'S2 path': [os.path.join(self.root_dir, self.paths[pdx]['S2'][cloudless_idx])],
                            },
                }
        
        return sample
    
    def __len__(self):
        # length of generated list
        return self.n_samples


if __name__ == "__main__":
    ##===================================================##
    ##===================================================##
    dataset = sen12mscrts(root='../../../Data/CR-sentinel/SEN12MSCR-TS/test',  
                          split='test', 
                          import_data_path='data/precomputed/sen12mscrts/splits/generic_3_test_all_s2cloudless_mask.npy',
                          min_cov=0.0,
                          max_cov=0.95,
                          cloud_masks='s2cloudless_mask',
                          import_csv_path='data/precomputed/sen12mscrts/pairs/test.csv')
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,shuffle=True)

    ##===================================================##
    _iter = 0
    for results in dataloader:
        cloudy_data = results['input']['S2'][0]
        cloudfree_data = results['target']['S2'][0]
        sar_data = results['input']['S1'][0]

        print('cloudy_data:', cloudy_data.shape)
        print('cloudfree_data', cloudfree_data.shape)
        print('sar_data', sar_data.shape)
        _iter += 1