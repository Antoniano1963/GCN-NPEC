import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from model import Model
# from model import AttentionModel
from baseline import RolloutBaseline
from data import generate_data, Generator
from config import Config, load_pkl, train_parser
import os


def train(cfg, log_path = None):
	torch.backends.cudnn.benchmark = True
	model = Model(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping, cfg.n_customer)
	# model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping)
	model.train()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	baseline = RolloutBaseline(model, cfg.task, cfg.weight_dir, cfg.n_rollout_samples, 
								cfg.embed_dim, cfg.n_customer, cfg.warmup_beta, cfg.wp_epochs, device, cfg.batch)
	optimizer = optim.Adam(model.parameters(), lr = cfg.lr)
	
	t1 = time()
	for epoch in range(cfg.epochs):
		ave_total_loss, ave_seq_loss, avg_cla_loss, avg_cost = 0., 0., 0.,  0.
		dataset = Generator(device, cfg.batch*cfg.batch_steps, cfg.n_customer)
		
		bs = baseline.eval_all(dataset)
		bs = bs.view(-1, cfg.batch) if bs is not None else None# bs: (cfg.batch_steps, cfg.batch) or None
		
		dataloader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)
		for t, inputs in enumerate(tqdm(dataloader)):
			
			# loss, L_mean = rein_loss(model, inputs, bs, t, device)
			L, ll, pi, groud_mat, pre_mat = model(inputs, decode_type='sampling')
# 			L, ll, pi, groud_mat, pre_mat = model(inputs, decode_type = 'sampling')
			predict_matrix = pre_mat.view(-1, 2).cuda()
			solution_matrix = groud_mat.view(-1).long().cuda()
			crossEntropy = nn.CrossEntropyLoss()
			classification_loss = crossEntropy(predict_matrix, solution_matrix)
			
			b = bs[t] if bs is not None else baseline.eval(inputs, L)
			se_loss, L_mean = ((L - b.to(device)) * ll).mean(), L.mean()

			loss = se_loss + classification_loss
			
			optimizer.zero_grad()
			loss.backward()
			# print('grad: ', model.Decoder.Wk1.weight.grad[0][0])
			# https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py
			nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type = 2)
			optimizer.step()
			
			ave_total_loss += loss.item()
			ave_seq_loss += se_loss.item()
			avg_cost += L_mean.item()
			avg_cla_loss += classification_loss
			
			if t%(cfg.batch_verbose) == 0:
				t2 = time()
				print('Epoch %d (batch = %d): ave_total_loss: %1.3f ave_seq_loss: %1.3f avg_cla_loss: %1.3f, seq_loss: %1.3f, class_loss: %1.3f, cost: %1.3f %dmin%dsec'%(
					epoch, t, ave_total_loss/(t+1), ave_seq_loss/(t+1), avg_cla_loss/(t+1), L_mean.item(), classification_loss, avg_cost, (t2-t1)//60, (t2-t1)%60))
				if cfg.islogger:
					if log_path is None:
						log_path = '%s%s_%s.csv' % (cfg.log_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
						with open(log_path, 'w') as f:
							f.write('time,epoch,batch,total_loss,seq_loss, class_loss, cost\n')
					with open(log_path, 'a') as f:
						f.write('Epoch %d (batch = %d): ave_total_loss: %1.3f ave_seq_loss: %1.3f avg_cla_loss: %1.3f, seq_loss: %1.3f, class_loss: %1.3f, cost: %1.3f %dmin%dsec'%(
					epoch, t, ave_total_loss/(t+1), ave_seq_loss/(t+1), avg_cla_loss/(t+1), L_mean.item(), classification_loss, avg_cost/(t+1), (t2-t1)//60, (t2-t1)%60))
				t1 = time()

		baseline.epoch_callback(model, epoch)
		if not os.path.exists('%s%s/%s' % (cfg.weight_dir, cfg.task, cfg.dump_date)):
			os.makedirs('%s%s/%s' % (cfg.weight_dir, cfg.task, cfg.dump_date))
		torch.save(model.state_dict(), '%s%s/%s/epoch%s.pt' % (cfg.weight_dir, cfg.task, cfg.dump_date, epoch))

if __name__ == '__main__':
	cfg = load_pkl(train_parser().path)
	train(cfg)	
