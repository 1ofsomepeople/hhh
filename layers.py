'''
Author: hushuwang hushuwang2019@ia.ac.cn
Date: 2022-07-06 15:05:33
LastEditors: hushuwang hushuwang2019@ia.ac.cn
LastEditTime: 2022-07-11 17:18:00
FilePath: /hushuwang/mahua/layers.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, hid_fea, num_nodes, num_fea):
        super(LSTM, self).__init__()
        """
            in_fea: cnn输入、lstm输入的维度
            hid_fea: lstm输出的维度
        """
        self.fc1 = nn.Linear(in_features=num_nodes*num_fea, out_features=hid_fea)
        self.lstm = nn.LSTM(input_size=hid_fea, hidden_size=hid_fea, batch_first=True)
        self.fc2 = nn.Linear(in_features=hid_fea, out_features=num_fea*num_nodes)

    def forward(self, edge_fea):
        """
            edge_fea: B*T*N*N
        """
        self.lstm.flatten_parameters()
        B,T,N,_ = edge_fea.shape
        edge_fea = edge_fea.reshape(B, T, -1)
        edge_fea = self.fc1(edge_fea)
        edge_fea = F.leaky_relu(edge_fea)
        edge_fea = self.lstm(edge_fea)[1][0]
        edge_fea = self.fc2(edge_fea)
        edge_fea = edge_fea.reshape(B, 1, N, -1)
        return edge_fea


class LR(nn.Module):
    def __init__(self, seq_len, num_nodes):
        super(LR, self).__init__()
        """
            seq_len:
            num_nodes:
        """
        self.W = nn.Parameter(torch.empty(size=(seq_len, num_nodes, num_nodes)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x):
        """
            x:B*T*N*N
        """
        x = torch.sum(x * self.W, dim=1, keepdim=True)
        return x


class HA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        return x


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, **kwargs) -> None:
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size, padding=self.kernel_size//2)
    def forward_step(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f*c_cur + i*g
        h_next = o*torch.tanh(c_next)
        return h_next, c_next
    def forward(self, in_seq):
        """
            in_seq:(B,T,N,N)
        """
        in_seq = torch.unsqueeze(in_seq, dim=2)
        B, T, _, H, W = in_seq.shape
        device = in_seq.device
        c_state = torch.zeros([B, self.hidden_dim, H, W]).to(device)
        h_state = torch.zeros([B, self.hidden_dim, H, W]).to(device)
        for i in range(T):
            h_state, c_state = self.forward_step(in_seq[:,i,:,:,:], h_state, c_state)
        h_state = torch.sum(h_state, dim=1, keepdim=True)
        return h_state


class GNNLSTM(nn.Module):
    """
        静态图GNN + LSTM
    """
    def __init__(self, input_dim, hidden_dim, dropout, alpha) -> None:
        super(GNNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_state_init = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.cell_state_init = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.gconv = GraphAttentionLayer(in_features=input_dim+hidden_dim, out_features=4*hidden_dim, dropout=dropout, alpha=alpha)
        self.decoder = nn.Linear(in_features=hidden_dim, out_features=input_dim)

    def forward_step(self, node_fea, adj, h_cur, c_cur):
        # adj = torch.where(node_fea>0.0, 1.0, 0.0) + adj # 自适应adj
        combined = torch.cat([node_fea, h_cur], dim=2)
        combined_conv = self.gconv(combined, adj)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=2)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f*c_cur + i*g
        h_next = o*torch.tanh(c_next)
        return h_next, c_next

    def forward(self, node_fea, adj):
        """
            node_fea: (B,T,N,C)
            edge_fea: (B,T,N,N)
        """
        B, T, N, C = node_fea.shape
        device = node_fea.device
        c_state = self.hidden_state_init(node_fea[:,0,:,:])
        h_state = self.cell_state_init(node_fea[:,0,:,:])
        # c_state = torch.zeros([B, N, self.hidden_dim]).to(device)
        # h_state = torch.zeros([B, N, self.hidden_dim]).to(device)
        multi_step_loss = 0.0
        for i in range(T):
            h_state, c_state = self.forward_step(node_fea[:,i,:,:], adj, h_state, c_state)
            if i<T-1:
                multi_step_loss += nn.MSELoss()(self.decoder(h_state), node_fea[:,i+1,:,:])
        h_state = self.decoder(h_state)
        h_state = torch.unsqueeze(h_state, dim=1)
        return h_state, multi_step_loss

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, training=True, **kwargs) -> None:
        super(MLPDecoder, self).__init__()
        self.dropout = dropout
        self.training = training
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.a = nn.Parameter(torch.empty(size=(2*self.in_dim, self.out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, inp):
        """
            inp:(B,N,C)
            todo: 实现3d版本
        """
        inp = F.dropout(inp, self.dropout, training=self.training)
        h1 = torch.mm(inp, self.a[:self.in_dim, :])
        h2 = torch.mm(inp, self.a[self.in_dim:, :])
        e = h1 + h2.T
        return e


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
            h: (B,N,C)
            adj: (N,N)
        """
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        Wh2 = torch.transpose(Wh2, dim0=1, dim1=2)
        # broadcast add
        e = Wh1 + Wh2
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'