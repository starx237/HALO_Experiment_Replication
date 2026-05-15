import argparse
from trainer import DynamicalModelTrainer
import torch
import pdb
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import yaml
from utils import save_latent_z3D, latent_scatter_3d, latent_scatter_2d, atanh
from dataset import MUG, Sprites
import functools
from utils import save_seq_img, video_transform

# 以下模块暂未使用
# from torchvision.utils import make_grid, save_image
# from torchvision import datasets
# from scipy.stats import entropy
# from matplotlib import cm
# from celluloid import Camera
# from sklearn.decomposition import PCA
# from evaluation import evaluation_scores
# from mugdataset import MUGDataset
# from video_dataset import VideoDataset


def test(config, trainer, test_loader, results_path, device):

	with torch.no_grad():
		print("Testing")
		#ssim, psnr, mse = evaluation_scores(test_loader, trainer, device)
		data_loader_iter = iter(test_loader)
		for j, batch in enumerate(test_loader, 0):
			print(f"Batch {j}")
			# Reconstruct Sequences
			# 兼容两种数据格式: Sprites 返回 tuple (data, labels)，MUG 返回 dict
			if isinstance(batch, (list, tuple)):
				data_test, labels = batch
			else:
				data_test = batch['images'].permute(0,2,1,3,4)
				labels = batch['categories']
			data_test = data_test.to(device[0])
			labels = labels.to(device[0])
			labels_onehot = torch.FloatTensor(labels.shape[0], trainer.config['model']['u_dim']).to(device[0])
			labels_onehot.zero_()
			labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
			batch_size, timesteps, channels, rows, columns = data_test.shape

			x_gen, x_recon, x_orig = trainer.generate(data_test, labels_onehot)
			#Save sequences
			for batch in range(batch_size):
				batch_data = torch.cat([x_orig[batch].unsqueeze(0), x_recon[batch].unsqueeze(0), x_gen[batch].unsqueeze(0)])
				batch_data = batch_data.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
				save_seq_img(batch_data,  f"{results_path}/reconstruct_seq_batch_{j}_sample_{batch}.png")
				batch_data = torch.cat([x_orig[batch].unsqueeze(0), x_gen[batch].unsqueeze(0)])
				batch_data = batch_data.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
				save_seq_img(batch_data,  f"{results_path}/generate_seq_batch_{j}_sample_{batch}.png")


			x_recon, image = trainer.image_to_seq(data_test, time=16)
			#Save sequences
			for batch in range(batch_size):
				batch_data = torch.cat([image[batch].unsqueeze(0), x_recon[batch]])
				batch_data = batch_data.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
				save_seq_img(batch_data,  f"{results_path}/image_to_seq_batch_{j}_sample_{batch}.png")
			
			x_recon, x_orig = trainer.reconstruct_all_actions(data_test)
			#Save sequences

			for batch in range(batch_size):
				batch_data = torch.cat([x_orig[batch], x_recon[batch]])
				batch_data = batch_data.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
				save_seq_img(batch_data,  f"{results_path}/seq_batch_{j}_sample_{batch}.png")
			
			batch2 = next(data_loader_iter)
			# 兼容两种数据格式
			if isinstance(batch2, (list, tuple)):
				data_test2, labels2 = batch2
			else:
				data_test2 = batch2['images'].permute(0,2,1,3,4)
				labels2 = batch2['categories']
			data_test2 = data_test2.to(device[0])
			labels2 = labels2.to(device[0])
			j += 1
			labels_onehot2 = torch.FloatTensor(labels2.shape[0], trainer.config['model']['u_dim']).to(device[0])
			labels_onehot2.zero_()
			labels_onehot2.scatter_(1, labels2.unsqueeze(1), 1)
			batch_size, timesteps, channels, rows, columns = data_test.shape

			x_11, x_22, x_12, x_21 = trainer.motion_composition(data_test, data_test2, labels_onehot, labels_onehot2)
			#Save sequences
			for sample in range(x_11.shape[0]):
				x_concat = torch.cat([data_test[sample].unsqueeze(0), data_test2[sample].unsqueeze(0), x_11[sample].unsqueeze(0), x_22[sample].unsqueeze(0), x_12[sample].unsqueeze(0), x_21[sample].unsqueeze(0)])
				x_concat = x_concat.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
				save_seq_img(x_concat, f"{results_path}/motion_composition_batch_{j}_{sample}.png")


			x_recon, x_orig = trainer.reconstruct(data_test, labels_onehot)
			#Save sequences
			for batch in range(batch_size):
				batch_data = torch.cat([x_orig[batch].unsqueeze(0), x_recon[batch].unsqueeze(0)])
				batch_data = batch_data.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
				save_seq_img(batch_data,  f"{results_path}/seq_batch_{j}_sample_{batch}.png")
			

			#Style Transfer
			x_11, x_22, x_12, x_21 = trainer.style_transfer(data_test, data_test2, labels_onehot, labels_onehot2)
			
			#Save sequences
			for sample in range(x_11.shape[0]):
				x_concat = torch.cat([data_test[sample].unsqueeze(0), data_test2[sample].unsqueeze(0), x_11[sample].unsqueeze(0), x_22[sample].unsqueeze(0), x_12[sample].unsqueeze(0), x_21[sample].unsqueeze(0)])
				x_concat = x_concat.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
				save_seq_img(x_concat, f"{results_path}/style_transfer_batch_{j}_{sample}.png")


			#Random Variant Sequence
			seq_sample = trainer.random_variant_sample(data_test, labels_onehot)
			batch, time, channels, rows, cols = seq_sample.shape
			seq_sample = seq_sample.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
			save_seq_img(seq_sample,  f"{results_path}/seq_rand_variant_batch_{j}.png")
			print(trainer.labels_dict)
			print(labels)

			#Random invariant sequence
			seq_sample = trainer.random_invariant_sample(data_test, labels_onehot)
			seq_sample = seq_sample.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
			save_seq_img(seq_sample,  f"{results_path}/seq_rand_invariant_batch_{j}.png")
			print(trainer.labels_dict)
			print(labels)


			#fixed invariant and sample all three subspaces
			seq_sample = trainer.random_all_variant_sample(data_test)
			for k, image in enumerate(seq_sample):
				nm_seq, time, channels, rows, cols = image.shape
				image = image.permute(0,1,3,4,2).mul(0.5).add(0.5).cpu().numpy()
				save_seq_img(image,  f"{results_path}/seq_rand_sample_{k}_batch_{j}.png")

			break  # 只测 1 个 batch 节省时间（取消此行可恢复全部）
	trainer.writer.close()


def get_config(config):
	with open(config, 'r') as f:
		return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/sprites.yaml', help='Configuration file')
parser.add_argument('--dataname', type=str, default='MUG', help='Configuration file')
opt = parser.parse_args()


config = get_config(opt.config)

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not config['cuda']:
	device = ['cpu']
else:
	device = config['gpu_ids']

results_path = f"{config['output_results']}_{config['data']}_TemporalLoss_{config['trainer']['temporalLoss']}_TempLossBeta_{config['trainer']['BetaT']}"\
		f"_condnSonU_{config['trainer']['model']['condnSonU']}_maskV_{config['trainer']['model']['projection']}"\
		f"_dynamics_{config['trainer']['model']['dynamics']}_qv_x_{config['trainer']['model']['qv_x']}"\
		f"_betaV_{config['trainer']['BetaV']}_useZ_{config['trainer']['model']['useZ']}_Zlstm_{config['trainer']['model']['uselstmZ']}"\
		f"_reconVloss_{config['trainer']['lossVrecon']}_reconxloss_{config['trainer']['reconloss']}"

model_path = f"{config['output_model']}_{config['data']}_TemporalLoss_{config['trainer']['temporalLoss']}_TempLossBeta_{config['trainer']['BetaT']}"\
		f"_condnsonU_{config['trainer']['model']['condnSonU']}_maskV_{config['trainer']['model']['projection']}"\
		f"_dynamics_{config['trainer']['model']['dynamics']}_qv_x_{config['trainer']['model']['qv_x']}"\
		f"_betaV_{config['trainer']['BetaV']}_useZ_{config['trainer']['model']['useZ']}_Zlstm_{config['trainer']['model']['uselstmZ']}"\
		f"_reconVloss_{config['trainer']['lossVrecon']}_reconxloss_{config['trainer']['reconloss']}"



ckpt = f"{model_path}/model_{config['test_epoch']}.pt"
state = torch.load(ckpt, map_location=torch.device('cpu'))
if not os.path.exists(results_path):
	os.makedirs(results_path)

if opt.dataname == 'MUG':
    assert 0, "MUG dataset path is not yet implemented (requires mugdataset/VideoDataset modules)"
    # 参考实现（需先实现 mugdataset / VideoDataset 模块）:
    # if config['trainer']['reconloss'] in ['l2', 'l1']:
    #     image_transforms = transforms.Compose([...])
    # dataset = MUGDataset(config['mug'], 'Test')
    # test_loader = DataLoader(VideoDataset(...), ...)

elif opt.dataname == 'sprites':
	dataset = Sprites(config['sprites'], train=False)
	config['sprites']['timesteps'] = dataset.timesteps
	config['trainer']['actions'] = len(dataset.labels_dict)
	config['trainer']['nm_seq'] = len(dataset.labels)
	test_loader = torch.utils.data.DataLoader(dataset, batch_size=config['trainer']['batch_size_test'], 
													shuffle=True, num_workers=4, drop_last=True)
	config['trainer']['model']['channels'] = dataset.channels
	config['trainer']['model']['width'] = dataset.rows
	config['trainer']['model']['height'] = dataset.columns
	config['trainer']['model']['u_dim'] = len(config['sprites']['actions'])




config['trainer']['model']['sequential'] = True
trainer = DynamicalModelTrainer(config['trainer'], config['data'], state['actions'], results_path, model_path, device)

trainer.vae.encoder.load_state_dict(state['encoder'])
trainer.vae.decoder.load_state_dict(state['decoder'])

if config['trainer']['model']['sequential'] and config['trainer']['model']['dynamics']!='Fourier':
	if 'lds' in state:
		trainer.vae.lds.load_state_dict(state['lds'])
	else:
		print("Warning: checkpoint does not contain 'lds' (probably saved during pretrain). Skipping lds load.")

test(config['trainer'], trainer, test_loader, results_path, device)
