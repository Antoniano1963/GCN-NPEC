import torch
import torch.nn as nn

from layers import MultiHeadAttention, DotProductAttention, AttentionPointer
from data import generate_data
from decoder_utils import TopKSampler, CategoricalSampler, Env
import numpy as np

class ClassificationDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super(ClassificationDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.sm = nn.Softmax(-1)

    def forward(self, x):
        e = x
        a = self.MLP(e)
        a = a.squeeze(-1)
        out = self.sm(a)
        return out

class DecoderCell(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, clip = 10., **kwargs):
		super().__init__(**kwargs)
		
		self.Wk1 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wk2 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_step = nn.Linear(embed_dim+1, embed_dim, bias = False)
		
		self.MHA = MultiHeadAttention(n_heads = n_heads, embed_dim = embed_dim, need_W = False)
		self.SHA = DotProductAttention(clip = clip, return_logits = True, head_depth = embed_dim)
		# SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
		self.env = Env

	def compute_static(self, node_embeddings, graph_embedding):
		#我才是把固定不动的元素计算出来，但是没看懂为啥这么多
		self.Q_fixed = self.Wq_fixed(graph_embedding[:,None,:])
		# [batch, 1, hidden_dum]
		self.K1 = self.Wk1(node_embeddings)
		self.V = self.Wv(node_embeddings)
		self.K2 = self.Wk2(node_embeddings)
		
	def compute_dynamic(self, mask, step_context):
		Q_step = self.Wq_step(step_context)
		# [batch, 1, hidden_dim]
		Q1 = self.Q_fixed + Q_step
		# [batch, 1, hidden_dim]
		#这里是公式16的W部分，为了简化计算，因为参数一样，一部分计算可以重复利用
		#这里的K和V是node节点的embbedding，这里参数相同所以可以服用
		Q2 = self.MHA([Q1, self.K1, self.V], mask = mask)
		# [batch, 1, hidden_dim]
		#用了自己定义的MHA,还要看，哭
		Q2 = self.Wout(Q2)
		logits = self.SHA([Q2, self.K2, None], mask = mask)
		# [batch, 1, node_num]
		# print(logits[0,0,:])
		#日了，这个模型和他说得也不是完全一样呀。。。
		return logits.squeeze(dim = 1)

	def forward(self, x, encoder_output, return_pi = False, decode_type = 'sampling'):
		# x 四部分组成， depot, demand, cur_position, edges
		#encoder_output 才是 encoder的节点embedding
		node_embeddings = encoder_output
		# node_embedding [batch, node_num. hidden_dim]
		graph_embedding = torch.mean(node_embeddings, dim = 1)
		#这个和am模型做的有啥区别吗，，， [batch, hidden+dim]
		self.compute_static(node_embeddings, graph_embedding)
		env = Env(x, node_embeddings)
		mask, step_context, D = env._create_t1()
		# mask [batch, node_num, 1]
		# step_context [batch, 1, hidden_dim + 1]?
		selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
		#这个选择器到时候再看吧
		log_ps, tours = [], []	
		for i in range(env.n_nodes*2):
			#这个函数里面应该是公式16
			logits = self.compute_dynamic(mask, step_context)
			log_p = torch.log_softmax(logits, dim = -1)
			# log_p [batch, node_num]
			next_node = selecter(log_p)
			# [batch, 1]
			mask, step_context, D = env._get_step(next_node, D)
			tours.append(next_node.squeeze(1))
			log_ps.append(log_p)
			if env.visited_customer.all():
				break

		pi = torch.stack(tours, 1)
		cost = env.get_costs(pi)
		ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi)
		
		if return_pi:
			return cost, ll, pi
		return cost, ll


class SequencialDecoder(nn.Module):
	def __init__(self, embed_dim=128, n_heads=8, clip=10., **kwargs):
		super().__init__(**kwargs)

		self.Wk1 = nn.Linear(embed_dim, embed_dim, bias=False)
		self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
		self.Wk2 = nn.Linear(embed_dim, embed_dim, bias=False)
		self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias=False)
		self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
		self.Wq_step = nn.Linear(embed_dim + 1, embed_dim, bias=False)
		self.gru = nn.GRU(embed_dim + 1, embed_dim, num_layers=2)
		self.pointer = AttentionPointer(embed_dim, use_tanh=True, use_cuda=True)
		self.embed_dim = embed_dim
		# SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
		self.env = Env
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x, encoder_output, return_pi=False, decode_type='sampling'):
		# x 四部分组成， depot, demand, cur_position, edges
		# encoder_output 才是 encoder的节点embedding
		node_embeddings = encoder_output
		batch_size = encoder_output.shape[0]
		# node_embedding [batch, node_num. hidden_dim]
		# 这个和am模型做的有啥区别吗，，， [batch, hidden+dim]
		# self.compute_static(node_embeddings, graph_embedding)
		env = Env(x, node_embeddings)
		mask, step_context, D = env._create_t1()
		hidden = torch.zeros((2, batch_size, self.embed_dim)).to(torch.device('cuda:0'))
		# mask [batch, node_num, 1]
		# step_context [batch, 1, hidden_dim + 1]?
		selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
		# 这个选择器到时候再看吧
		log_ps, tours = [], []
		for i in range(env.n_nodes * 2):
			# 这个函数里面应该是公式16
			last_x = step_context.permute(1, 0, 2)
			_, hidden = self.gru(last_x, hidden)
			z = hidden[-1]
			_, u = self.pointer(z, node_embeddings.permute(1, 0, 2))
			new_mask = mask.squeeze(-1)
			u = u.masked_fill_(new_mask, -np.inf)
			# probs = self.softmax(u)
			log_p = self.softmax(u)
			# log_p [batch, node_num]
			next_node = selecter(log_p)
			# [batch, 1]
			mask, step_context, D = env._get_step(next_node, D)
			tours.append(next_node.squeeze(1))
			log_ps.append(log_p)
			if env.visited_customer.all():
				break

		pi = torch.stack(tours, 1)
		cost = env.get_costs(pi)
		ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi)

		if return_pi:
			return cost, ll, pi
		return cost, ll

if __name__ == '__main__':
	batch, n_nodes, embed_dim = 5, 21, 128
	data = generate_data(n_samples = batch, n_customer = n_nodes-1)
	decoder = DecoderCell(embed_dim, n_heads = 8, clip = 10.)
	node_embeddings = torch.rand((batch, n_nodes, embed_dim), dtype = torch.float)
	graph_embedding = torch.rand((batch, embed_dim), dtype = torch.float)
	encoder_output = (node_embeddings, graph_embedding)
	# a = graph_embedding[:,None,:].expand(batch, 7, embed_dim)
	# a = graph_embedding[:,None,:].repeat(1, 7, 1)
	# print(a.size())

	decoder.train()
	cost, ll, pi = decoder(data, encoder_output, return_pi = True, decode_type = 'sampling')
	print('\ncost: ', cost.size(), cost)
	print('\nll: ', ll.size(), ll)
	print('\npi: ', pi.size(), pi)

	# cnt = 0
	# for i, k in decoder.state_dict().items():
	# 	print(i, k.size(), torch.numel(k))
	# 	cnt += torch.numel(k)
	# print(cnt)

	# ll.mean().backward()
	# print(decoder.Wk1.weight.grad)
	# https://discuss.pytorch.org/t/model-param-grad-is-none-how-to-debug/52634	