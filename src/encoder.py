import torch
import torch.nn as nn
# from torchsummary import summary

from layers import MultiHeadAttention
from data import generate_data
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GCNLayer(nn.Module):
    def __init__(self,hidden_dim):
        super(GCNLayer,self).__init__()
        # node GCN Layers
        self.W_node = nn.Linear(hidden_dim, hidden_dim)
        self.V_node_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_node = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = 1)
        self.Relu=nn.ReLU()
        self.Ln1_node = nn.LayerNorm(hidden_dim)
        self.Ln2_node =nn.LayerNorm(hidden_dim)

        # edge GCN Layers
        self.W_edge = nn.Linear(hidden_dim,hidden_dim)
        self.V_edge_in = nn.Linear(hidden_dim, hidden_dim)
        self.V_edge = nn.Linear(2 * hidden_dim,hidden_dim)
        self.W1_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W2_edge = nn.Linear(hidden_dim, hidden_dim)
        self.W3_edge =nn.Linear(hidden_dim,hidden_dim)
        self.Relu =nn.ReLU()
        self.Ln1_edge = nn.LayerNorm(hidden_dim)
        self.Ln2_edge = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
 
    def forward(self, x, e,neighbor_index):
        #GCN中每一层做的事情
        # 其中x是node embedding e 是 edge embedding
        # node embedding
        # x 是 node embedding [batch, node_number, dim_node]
        batch_size=x.size(0)
        node_num =x.size(1)
        node_hidden_dim =x.size(-1)
        t=x.unsqueeze(1).repeat(1,node_num,1,1)
        # t [batch node_number, nodenumber, dim_node]
#         print(neighbor_index.shape)
#         print(neighbor_index[1])
        neighbor_index = neighbor_index.unsqueeze(3).repeat(1,1,1,node_hidden_dim)
        # neighbour_index [batch, nodenumber, k, hidden_dim]
        # print("neighbor_index.shape: " + str(neighbor_index.shape))
#         print(neighbor_index[1])
        neighbor = t.gather(2, neighbor_index)
        #gather函数就是用来去除k近邻的，但是用的方法比较取巧，反正我没看懂，真的这块出错了再看吧
        #根据 neighbor_index 去取K近邻
        # print("neighbor.shape:" + str(neighbor.shape))
        # neighbor [batch, node_num, k, hidden]
        neighbor = neighbor.view(batch_size, node_num,-1,node_hidden_dim)

        # print("neighbour shape: view: " + str(neighbor.shape))
        #x是key， neighbour 是Key和value 使用了nn的多头注意力模型，但是是当一个注意力层使用的
        # x [batch, node_num, node_dim]
        att, _ = self.attn(x, neighbor, neighbor)
        # att [batch, node_num, node_dim]
        # print("att shape:" + str(att.shape))
        #返回的是 attention后的值 和 attention的权重 att [batch, node_num, hidden_dim]
        out = self.W_node(att)
        #对应公式9
        # out [batch, node_num, hidden_dim]
        h_nb_node = self.Ln1_node(x + self.Relu(out))
        #加入skip-connection 残差连接
        # 这里 GCN 聚合子层完成
        # hN(i) = h_nb_node [batch, node_num. hidden_dim]
        #self.V_node_in(x) [batch, node_num, hidden_Dim]
        # self.V_edge 把 2 * hidden_dim 映射回 hidden_dim
        #这个relu要不要加，要看看另外一个模型咯
        #最后是残差链接
        #h_node [batch, node_num, hidden_dim]

        h_node=self.Ln2_node(h_nb_node+self.Relu(self.V_node(torch.cat([self.V_node_in(x),h_nb_node], dim=-1))))
        #公式12 组合计算出的h_nb_node（包含残差），和之前的输入，通过cat聚合后通过Linear返回之前的维度

        # edge embedding
        x_from =x.unsqueeze(2).repeat(1,1,node_num,1)
        # print("x_from shape: "+ str(x_from.shape))
        #x_from [batch, node_num, node_num, hidden_num]

        x_to=x.unsqueeze(1).repeat(1, node_num, 1,1)
        # print("x_to shape: " + str(x_to.shape))
        #x_to [batch, node_num, node_num, hidden_num]
        h_nb_edge =self.Ln1_edge(e+self.Relu(self.W_edge(self.W1_edge(e)+self.W2_edge(x_from)+self.W3_edge(x_to))))
        # W1_edge 对应公式 10 第一项
        # W2_edge 对应公式 10 第二项
        # W3_edge 对应公式 10 第三项
        # 最后 w_edge 对应 10 中的 WEl
        # 最后残差链接
        # print("h_nb_edge shape: " + str(h_nb_edge.shape))
        #残差连接在这一步，公式10，11
        h_edge =self.Ln2_edge(h_nb_edge + self.Relu(self.V_edge(torch.cat([self.V_edge_in(e), h_nb_edge],dim=-1))))
        #layernorm在这一步，公式13 和之前一样
        #那么这里GCN的问题就是 K 近邻怎么求的 以及RELU要不要加

        
        return h_node, h_edge
    
    
class GCN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, node_hidden_dim, edge_hidden_dim, gcn_num_layers, k=10):
        super(GCN, self).__init__()
        self.k = k
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.gcn_num_layers = gcn_num_layers

        self.node_W1 = nn.Linear(2, self.node_hidden_dim)
        self.node_W2 = nn.Linear(2, self.node_hidden_dim // 2)
        self.node_W3 = nn.Linear(1, self.node_hidden_dim // 2)
        self.edge_W4 = nn.Linear(1, self.edge_hidden_dim // 2)
        self.edge_W5 = nn.Linear(1, self.edge_hidden_dim // 2)
        self.nodes_embedding = nn.Linear(
            self.node_hidden_dim, self.node_hidden_dim, bias=True)
        self.edges_embedding = nn.Linear(
            self.edge_hidden_dim, self.edge_hidden_dim, bias=True)
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(self.node_hidden_dim) for i in range(self.gcn_num_layers)])
        self.Relu = nn.ReLU()

    def forward(self,pack):
        de, cus, demand, dis = pack
        # dis 是距离矩阵 0 是depot
        # cus 是所有客户的坐标 [batch, node_num, 2]
        # de 存储的是depot 的坐标 [batch, 2]
        # demand 记录节点的需求 [batch, node_num]
#         print(de.shape)
#         print(cus.shape)
        node = torch.cat([de.unsqueeze(-2),cus],dim = 1)
        batch_size =node.size(0)
        # print("dis shape:"+ str(dis.shape))
#         print(node.shape)
#         print(batch_size)
        node_num = node.size(1)
#         print(node_num)
        # node=torch.cat([node,timewin],dim=2)
        # edge =torch.cat([dis.unsqueeze(3),timedis.unsqueeze(3)], dim=3)
        '''
        edge = dis.unsqueeze(3).cuda()
        '''
        edge = dis.unsqueeze(3).to(device)
#         device = torch.device('cuda:0')
        '''
        self_edge=torch.arange(0,node_num).unsqueeze(0).repeat(batch_size,1).unsqueeze(2).cuda()
        '''
        self_edge = torch.arange(0, node_num).unsqueeze(0).repeat(batch_size, 1).unsqueeze(2).to(device)
        # self_dege [batch, node_num, 1]
#         print(self_edge.shape)
        order=dis.sort(2)[1]
        #这里取的是index!
#         print('order', order.shape)
        '''
        neighbor_index=order[:,:,1:self.k+1].cuda()
        '''
        neighbor_index = order[:, :, 1:self.k + 1].to(device)
        #应该是获得十个最近的邻居的 index

        with torch.no_grad():
            '''
            a=torch.zeros_like(dis).cuda()
            '''
            a = torch.zeros_like(dis).to(device)
            a=torch.scatter(a,2,neighbor_index,1)
            a=torch.scatter(a,2,self_edge,-1)
        #生成邻接矩阵A,公式3 a [batch, node_num + 1 , node_num + 1]
        '''
        depot=node[:,0,:].cuda()
        '''
        depot = node[:, 0, :].to(device)
        # 这个depot 和 de 似乎是一样的....，竟然没有保留维度
        dedmand = torch.zeros((1))
        '''
        demand=demand.unsqueeze(2).cuda()
        customer =node[:,1:,].cuda()
        '''
        demand=demand.unsqueeze(2).to(device)
        customer =node[:,1:,].to(device)
#         print(depot.shape, demand.shape, customer.shape)
        # Node and edge embedding
#         print(depot)
        depot_embedding=self.Relu(self.node_W1(depot))
#         print(self.node_W2(customer).shape, self.node_W3(demand).shape)
        #depot_embedding [batch, hidden_dim]
        customer_embedding=self.Relu(torch.cat([self.node_W2(customer),self.node_W3(demand)],dim=2))
        #对节点特征的非线性层处理
        # customer [batch, node_num. hidden_dim]
        x=torch.cat([depot_embedding.unsqueeze(1),customer_embedding],dim=1)
        #对depot的非线性层处理，公式2
        # x [batch, node_num + 1 , hidden_dim]

#         print(edge.shape, a.shape)
        e=self.Relu(torch.cat([self.edge_W4(edge),self.edge_W5(a.unsqueeze(3))],dim=3))
        #对边的非线性层处理，公式4

        x=self.nodes_embedding(x)
        e=self.edges_embedding(e)
        # print("edge_embedding" + str(e.shape))
        #公式5，6
        # x node_embedding [batch, node_number, node_dim]
        # e edge_embedding [batch, node_number, node_number, node_dim]
        #将点和边的特征都映射到dh维
        for layer in range(self.gcn_num_layers):
            x,e =self.gcn_layers[layer](x,e,neighbor_index) #BxVxH,BxVxVxH
            # 处理好的输入进入N的GCN的层进行信息汇聚
            # x 处理后的node_embedding
            # e 处理后的edge_embedding
            # print("e shape out GCN" + str(e.shape))

        return x, e
    



# if __name__ == '__main__':
#     batch = 5
#     n_nodes = 21
#     encoder = GraphAttentionEncoder(n_layers = 1)
#     data = generate_data(n_samples = batch, n_customer = n_nodes-1)
#     # mask = torch.zeros((batch, n_nodes, 1), dtype = bool)
#     output = encoder(data, mask = None)
#     print('output[0].shape:', output[0].size())
#     print('output[1].shape', output[1].size())
#
#     # summary(encoder, [(2), (20,2), (20)])
#     cnt = 0
#     for i, k in encoder.state_dict().items():
#         print(i, k.size(), torch.numel(k))
#         cnt += torch.numel(k)
#     print(cnt)

	# output[0].mean().backward()
	# print(encoder.init_W_depot.weight.grad)

