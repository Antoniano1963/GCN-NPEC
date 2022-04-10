import torch
import torch.nn as nn
import math

class DotProductAttention(nn.Module):
	def __init__(self, clip = None, return_logits = False, head_depth = 16, inf = 1e+10, **kwargs):
		super().__init__(**kwargs)
		self.clip = clip
		self.return_logits = return_logits
		self.inf = inf
		self.scale = math.sqrt(head_depth)
		# self.tanh = nn.Tanh() 

	def forward(self, x, mask = None):
		""" Q: (batch, n_heads, q_seq(=n_nodes or =1), head_depth)
			K: (batch, n_heads, k_seq(=n_nodes), head_depth)
			logits: (batch, n_heads, q_seq(this could be 1), k_seq)
			mask: (batch, n_nodes, 1), e.g. tf.Tensor([[ True], [ True], [False]])
			mask[:,None,None,:,0]: (batch, 1, 1, n_nodes) ==> broadcast depending on logits shape
			[True] -> [1 * -np.inf], [False] -> [logits]
			K.transpose(-1,-2).size() == K.permute(0,1,-1,-2).size()
		"""
		Q, K, V = x
		logits = torch.matmul(Q, K.transpose(-1,-2)) / self.scale
		if self.clip is not None:
			logits = self.clip * torch.tanh(logits)
			
		if self.return_logits:
			if mask is not None:
				return logits.masked_fill(mask.permute(0,2,1) == True, -self.inf)
			return logits

		if mask is not None:
			logits = logits.masked_fill(mask[:,None,None,:,0].repeat(1,logits.size(1),1,1) == True, -self.inf)
			
		probs = torch.softmax(logits, dim = -1)
		return torch.matmul(probs, V)

class MultiHeadAttention(nn.Module):
	def __init__(self, n_heads = 8, embed_dim = 128, clip = None, return_logits = None, need_W = None):
		super().__init__()
		self.n_heads = n_heads
		self.embed_dim = embed_dim
		self.head_depth = self.embed_dim // self.n_heads
		if self.embed_dim % self.n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")
		
		self.need_W = need_W 
		self.attention = DotProductAttention(clip = clip, return_logits = return_logits, head_depth = self.head_depth)
		if self.need_W:
			self.Wk = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wq = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.init_parameters()
	
	def init_parameters(self):
		for name, param in self.named_parameters():
			if name == 'Wout.weight':
				stdv = 1. / math.sqrt(param.size(-1))
			elif name in ['Wk.weight', 'Wv.weight', 'Wq.weight']:
				stdv = 1. / math.sqrt(self.head_depth)
			else:
				raise ValueError
			param.data.uniform_(-stdv, stdv)

	def split_heads(self, T):
		""" https://qiita.com/halhorn/items/c91497522be27bde17ce
			T: (batch, n_nodes, self.embed_dim)
			T reshaped: (batch, n_nodes, self.n_heads, self.head_depth)
			return: (batch, self.n_heads, n_nodes, self.head_depth)
			
			https://raishi12.hatenablog.com/entry/2020/04/20/221905
		"""
		shape = T.size()[:-1] + (self.n_heads, self.head_depth)
		T = T.view(*shape)
		return T.permute(0,2,1,3)

	def combine_heads(self, T):
		""" T: (batch, self.n_heads, n_nodes, self.head_depth)
			T transposed: (batch, n_nodes, self.n_heads, self.head_depth)
			return: (batch, n_nodes, self.embed_dim)
		"""
		T = T.permute(0,2,1,3).contiguous()
		shape = T.size()[:-2] + (self.embed_dim, )
		return T.view(*shape)

	def forward(self, x, mask = None):
		"""	q, k, v = x
			encoder arg x: [x, x, x]
			shape of q: (batch, n_nodes, embed_dim)
			output[0] - output[h_heads-1]: (batch, n_nodes, head_depth)
			--> concat output: (batch, n_nodes, head_depth * h_heads)
			return output: (batch, n_nodes, embed_dim)
		"""
		Q, K, V = x
		if self.need_W:
			Q, K, V = self.Wq(Q), self.Wk(K), self.Wv(V)
		Q, K, V = list(map(self.split_heads, [Q, K, V]))
		output = self.attention([Q, K, V], mask = mask)
		#这里它使用自定义的函数实现了公式16
		output = self.combine_heads(output)
		if self.need_W:
			return self.Wout(output)
		return output


class AttentionPointer(nn.Module):
	def __init__(self, hidden_dim, use_tanh=False, use_cuda=False):
		super(AttentionPointer, self).__init__()
		self.hidden_dim = hidden_dim
		self.use_tanh = use_tanh

		self.project_hidden = nn.Linear(hidden_dim, hidden_dim)
		self.project_x = nn.Conv1d(hidden_dim, hidden_dim, 1, 1)
		self.C = 10
		self.tanh = nn.Tanh()

		v = torch.FloatTensor(hidden_dim)
		if use_cuda:
			v = v.cuda()
		self.v = nn.Parameter(v)
		self.v.data.uniform_(-(1. / math.sqrt(hidden_dim)) , 1. / math.sqrt(hidden_dim))

	def forward(self, hidden, x):
		'''
		@param hidden: (batch_size, hidden_dim)
		@param x: (node_num, batch_size, hidden_dim)
		'''
		x = x.permute(1, 2, 0)
		q = self.project_hidden(hidden).unsqueeze(2)  # batch_size x hidden_dim x 1
		e = self.project_x(x)  # batch_size x hidden_dim x node_num
		# expand the hidden by node_num
		# batch_size x hidden_dim x node_num
		expanded_q = q.repeat(1, 1, e.size(2))
		# batch x 1 x hidden_dim
		v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)
		# (batch_size x 1 x hidden_dim) * (batch_size x hidden_dim x node_num)
		u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
		if self.use_tanh:
			logits = self.C * self.tanh(u)
		else:
			logits = u
		return e, logits

if __name__ == '__main__':
	mha = MultiHeadAttention(n_heads = 8, embed_dim = 128, need_W = True)
	batch, n_nodes, embed_dim = 5, 21, 128
	# x = torch.randn((batch, n_nodes, embed_dim))
	x = torch.randn((batch, n_nodes, embed_dim), dtype = torch.float)
	mask = torch.zeros((batch, n_nodes, 1), dtype = torch.bool)
	output = mha([x,x,x], mask = mask)
	print('output.size()', output.size())


