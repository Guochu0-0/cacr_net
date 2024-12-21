import torch
import time
import os
import json
import datetime
import numpy as np
from tqdm import tqdm
from metrics import PSNR
from tensorboardX import SummaryWriter

class Generic_Train():
	def __init__(self, model, config, train_dataloader=None, val_dataloader=None, test_dataloader=None):
		self.model=model
		self.config=config
		self.train_dataloader=train_dataloader
		self.val_dataloader=val_dataloader
		self.test_dataloader=test_dataloader

		if train_dataloader:
			current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			exp_name_with_time = f"{config.EXP_NAME}_{current_time}"
			self.writer = SummaryWriter(os.path.join('runs', exp_name_with_time))


	def train(self):
		
		total_steps = 0
		log_loss = 0
		best_score = 0

		for epoch in range(self.config.EPOCH):

			train_psnr = 0

			time_1 = time.time()
			for i, batch in enumerate(tqdm(self.train_dataloader)):
				total_steps+=1

				time_2 = time.time()
				time_load = time_2 - time_1
				print(f"time_load: {time_load}s")

				self.model.set_input(batch)
				batch_loss = self.model.optimize_parameters()
				log_loss = log_loss + batch_loss

				train_psnr += PSNR(self.model.pred_cloudfree_data, self.model.cloudfree_data) 

				if total_steps % self.config.LOG_ITER == 0:

					avg_log_loss = log_loss/self.config.LOG_ITER

					self.writer.add_scalar('batch_loss', avg_log_loss, total_steps)

					log_loss = 0
				
				time_1 = time.time()
				time_process = time_1 - time_2
				print(f"time_process: {time_process}s")

			self.writer.add_scalar('train_psnr', train_psnr/len(self.train_dataloader), epoch)

			if (epoch+1) % self.config.VAL_FREQ == 0:

				print("validation...")
				score = self.val(epoch)
				self.writer.add_scalar('val_psnr', score, epoch)

				if score > best_score:  # save best model
					best_score = score
					self.model.save_checkpoint('best')

			if epoch >= self.config.EPOCH - 10:
				print("test...")
				score = self.val_test(epoch)
				self.writer.add_scalar('test_psnr', score, epoch)

			# if self.model.scheduler_G:
			# 	self.model.scheduler_G.step()
			
			if epoch % self.config.SAVE_FREQ == 0:
				self.model.save_checkpoint(epoch)


	def val(self, epoch):
		self.model.net_G.eval()

		with torch.no_grad():

			_iter = 0
			score = 0

			for idx, data in enumerate(tqdm(self.val_dataloader)):
				self.model.set_input(data)
				score += self.model.val_scores()['PSNR']
				_iter += 1

				# if _iter%2 == 0:
				# 	self.model.val_img_save(epoch, idx=idx)

			score = score/_iter
		
		self.model.net_G.train()

		return score
	
	def val_test(self, epoch):
		self.model.net_G.eval()

		with torch.no_grad():

			_iter = 0
			score = 0

			for idx, data in enumerate(tqdm(self.test_dataloader)):
				self.model.set_input(data)
				score += self.model.val_scores()['PSNR']
				_iter += 1

				# if _iter%2 == 0:
				# 	self.model.val_img_save(epoch, idx=idx)

			score = score/_iter
		
		self.model.net_G.train()

		return score

	def test(self):
		iters = 0
		PSNR_4 = 0
		SSIM_4 = 0
		SAM_4 = 0
		MAE_4 = 0

		results = []
		# self.model.load_checkpoint('best')
		self.model.load_checkpoint(self.config.ckpt_epnum)
		self.model.net_G.eval()
		with torch.no_grad():
			for idx, inputs in enumerate(tqdm(self.test_dataloader)):
				if inputs == {}:
					iters += 1
					continue
				self.model.set_input(inputs)

				results_dict = self.model.val_scores()
				# self.model.val_img_save(epoch='test', idx=idx)

				psnr_4, ssim_4, angles, pixel_errors = results_dict['PSNR'], results_dict['SSIM'], results_dict['SAM'], results_dict['MAE']
				mae_4 = pixel_errors.mean()
				sam_4 = angles.mean()

				PSNR_4 += psnr_4
				SSIM_4 += ssim_4
				SAM_4 += sam_4
				MAE_4 += mae_4

				# mean_error = inputs['statistics']['Mean error'][0].item()
				mean_cov = inputs['statistics']['Mean cov'][0].item()

				results.append({
					'iteration': iters,
					'cloudy_name': self.model.cloudy_name,
					'PSNR': psnr_4,
					'SSIM': ssim_4,
					'SAM': float(sam_4),
					'MAE': float(mae_4),
					'mean_cov': mean_cov,
					# 'mean_error': mean_error
				})
				print(f'{iters}: PSNR:{psnr_4:.4f}, SSIM:{ssim_4:.4f}, SAM:{sam_4:.4f}, MAE:{mae_4:.4f}')

				iters += 1

		final_results = {
			'Average_PSNR': PSNR_4 / iters,
			'Average_SSIM': SSIM_4 / iters,
			'Average_SAM': SAM_4 / iters,
			'Average_MAE': MAE_4 / iters
		}
		
		print('Testing done.')
		print(f"PSNR: {final_results['Average_PSNR']:.4f}, SSIM: {final_results['Average_SSIM']:.4f}, "
			f"SAM: {final_results['Average_SAM']:.4f}, MAE: {final_results['Average_MAE']:.4f}")

		save_data = {
			'per_iteration': results,
			'final_results': final_results
		}

		save_dir = os.path.join('results', self.config.EXP_NAME)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		with open(os.path.join(save_dir, 'results.json'), 'w') as f:
			json.dump(save_data, f, indent=4)

