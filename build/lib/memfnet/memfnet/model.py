import sys
from sympy import cycle_length
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.autograd import Function
from memfnet.segments import *

# These functions are based on the implementation from roost by Rhys E. A. Goodall & Alpha A. Lee
# Source: https://github.com/CompRhys/roost

class MEMFNet(nn.Module):
    """
    predict the capacity based on input information

    residue network design for cycling prediction
    """

    def __init__(
        self,
        elem_emb_len,
        elem_fea_len = 32,
        vol_fea_len = 64,
        rate_fea_len = 16,
        cycle_fea_len = 16,
        sin_fea_len=16,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        out_hidden=[256, 128, 64],
        weight_pow = 1,
        activation = nn.ReLU,
        mu = 0,
        std = 1,
        batchnorm_graph = False,
        batchnorm_condition = False,
        batchnorm_mix = False,
        batchnorm_main = False,
        **kwargs
    ):
        if isinstance(out_hidden[0], list):
            raise ValueError("boo hiss bad user")
            # assert all([isinstance(x, list) for x in out_hidden]),
            #   'all elements of out_hidden must be ints or all lists'
            # assert len(out_hidden) == len(n_targets),
            #   'out_hidden-n_targets length mismatch'

        super().__init__()

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
            "weight_pow": weight_pow,
            "activation": activation,
            "batchnorm": batchnorm_graph

        }

        self.mu = mu
        self.std = std

        # 添加两个新的嵌入层用于煅烧条件
        self.sin1_temp_embedding = build_mlp(input_dim=1, output_dim=sin_fea_len, 
                                        hidden_dim=2*sin_fea_len, activation=activation, batchnorm=batchnorm_condition)
        self.sin1_time_embedding = build_mlp(input_dim=1, output_dim=sin_fea_len, 
                                        hidden_dim=2*sin_fea_len, activation=activation, batchnorm=batchnorm_condition)
        self.sin1_cond_embedding = build_mlp(input_dim=2, output_dim=sin_fea_len, 
                                        hidden_dim=2*sin_fea_len, activation=activation, batchnorm=batchnorm_condition)
        self.sin1_fea_embedding = build_mlp(input_dim=64, output_dim=sin_fea_len, # input_dim=32
                                        hidden_dim=2*sin_fea_len, activation=activation, batchnorm=batchnorm_condition)
        
        self.sin2_temp_embedding = build_mlp(input_dim=1, output_dim=sin_fea_len, 
                                        hidden_dim=2*sin_fea_len, activation=activation, batchnorm=batchnorm_condition)
        self.sin2_time_embedding = build_mlp(input_dim=1, output_dim=sin_fea_len, 
                                        hidden_dim=2*sin_fea_len, activation=activation, batchnorm=batchnorm_condition)
        self.sin2_cond_embedding = build_mlp(input_dim=2, output_dim=sin_fea_len, 
                                        hidden_dim=2*sin_fea_len, activation=activation, batchnorm=batchnorm_condition)
        self.sin2_fea_embedding = build_mlp(input_dim=64, output_dim=sin_fea_len, # input_dim=32
                                        hidden_dim=2*sin_fea_len, activation=activation, batchnorm=batchnorm_condition)
        
        self.sin_fea_embedding = build_mlp(input_dim=32, output_dim=sin_fea_len, 
                                        hidden_dim=2*sin_fea_len, activation=activation, batchnorm=batchnorm_condition)
        self.sin2_embedding = build_mlp(input_dim=2, output_dim=16, hidden_dim=2*16, activation=activation, batchnorm=batchnorm_condition)


        self.rate_embedding = build_mlp(input_dim= 1, output_dim= rate_fea_len,
                                        hidden_dim = 2 * rate_fea_len,
                                        activation= activation, batchnorm= batchnorm_condition)

        self.cycle_embedding = build_mlp(input_dim= 1, output_dim= cycle_fea_len,
                                         hidden_dim = 2 * cycle_fea_len,
                                         activation= activation, batchnorm= batchnorm_condition)
        
        self.sin_embedding = build_mlp(input_dim= 64, output_dim= 32,
                                         hidden_dim = 2 * 64,
                                         activation= activation, batchnorm= batchnorm_condition)
        
        self.embedding = build_mlp(input_dim= 232, output_dim= 200,
                                         hidden_dim = 2 * 64,
                                         activation= activation, batchnorm= batchnorm_condition)
     

        self.material_nn = DescriptorNetwork(**desc_dict)

        # Step 3: Define gate layers for each combination
        self.encode_sin = build_gate(input_dim=64, output_dim=elem_fea_len, activation=activation, batchnorm=batchnorm_mix)
        self.encode_comp_sin = build_gate(input_dim=64, output_dim=elem_fea_len, activation=activation, batchnorm=batchnorm_mix)


        self.encode_rate = build_gate(input_dim=48, output_dim=elem_fea_len, activation=activation, batchnorm=batchnorm_mix)
        self.encode_cycle = build_gate(input_dim=48, output_dim=elem_fea_len, activation=activation, batchnorm=batchnorm_mix)

        self.delta_N = nn.Linear(1, elem_fea_len, bias= False)

        self.encode_voltage = EncodeVoltage(hidden_dim = vol_fea_len,
                                            output_dim = elem_fea_len,
                                            activation= activation,
                                            batchnorm= batchnorm_main)

        self.add_voltage = forwardVoltage(input_dim = elem_fea_len,
                                           output_dim = elem_fea_len,
                                           activation= nn.Softplus,
                                           batchnorm= batchnorm_main)



        self.fc = build_mlp(input_dim= elem_fea_len , output_dim= 1,
                            hidden_dim = elem_fea_len, activation= nn.Softplus,
                            batchnorm= batchnorm_main)


        self.elem_fea_attn = CrossAttentionLayer(embedding_dim=64)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        # 添加用于存储中间特征的变量
        self.intermediate_features = {}

    def get_intermediate_features(self):
        """返回模型的中间特征"""
        return self.intermediate_features
        
    def clear_intermediate_features(self):
        """清除存储的中间特征"""
        self.intermediate_features = {}

    def forward(self,
                elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx,
                V_window, rate, cycle, Vii,
                expand_cycle,
                sin1_temp, sin1_time, sin2_temp, sin2_time,
                expand_vii,
                return_direct = True,
                return_feature = False):
        """
        Forward pass through the material_nn and output_nn
        """
        # Step 1: 存储初始元素特征
        self.intermediate_features['initial_elem_fea'] = elem_fea.detach()

        # 在关键点添加特征存储
        sin1_fea = self.sin1_cond_embedding(torch.cat([sin1_temp, sin1_time], dim=1))
        sin2_fea = self.sin2_cond_embedding(torch.cat([sin2_temp, sin2_time], dim=1))
        
        cond_sin, attn_sin = self.encode_sin(torch.cat([sin1_fea, sin2_fea], dim=1))
        self.intermediate_features['synthesis_attention'] = attn_sin.detach()
        
        elem_fea_synthesis = torch.cat([elem_fea, cond_sin], dim=1)
        elem_fea_synthesis = self.embedding(elem_fea_synthesis)
        
        # 存储循环注意力特征
        elem_fea_t, elem_fea_t_attn = self.elem_fea_attn(elem_fea_synthesis, elem_weights, expand_cycle, expand_vii)
        self.intermediate_features['cycle_attention'] = elem_fea_t.detach()
        # self.intermediate_features['cycle_attention'] = elem_fea_t_attn
        
        # 存储GNN更新后的特征
        crys_fea, elem_fea_updated = self.material_nn(
            elem_weights, elem_fea_t, self_fea_idx, nbr_fea_idx, cry_elem_idx,
        )
        self.intermediate_features['gnn_features'] = elem_fea_updated.detach()
        
        # 存储电压相关特征
        rate_fea = self.rate_embedding(rate)
        cycle_fea = self.cycle_embedding(cycle)
        
        cond_rate, attn_rate = self.encode_rate(torch.cat([crys_fea, rate_fea], dim=1))
        self.intermediate_features['rate_attention'] = attn_rate.detach()
        
        cond_cycle, attn_cycle = self.encode_cycle(torch.cat([cond_rate, cycle_fea], dim=1))
        self.intermediate_features['final_cycle_attention'] = attn_cycle.detach()
        
        # 电压编码和最终预测
        x_vol, _ = self.encode_voltage(V_window, Vii)
        x = self.add_voltage(x_vol, cond_rate)
        self.intermediate_features['voltage_features'] = x.detach()
        
        q_out = self.softplus(self.fc(x))

        if return_direct:
            return q_out
        elif return_feature:
            return (crys_fea, cond_rate, cond_cycle)
        else:
            # Create dummy gradients if Vii or elem_weights weren't used
            if not Vii.requires_grad:
                Vii.requires_grad = True
            
            # Compute gradients with allow_unused=True
            q_grad = torch.autograd.grad(q_out, Vii, 
                                    grad_outputs=torch.ones_like(q_out),
                                    create_graph=True, 
                                    retain_graph=True,
                                    allow_unused=True)[0]
            
            grad_weights = torch.autograd.grad(q_out, elem_weights, 
                                            grad_outputs=torch.ones_like(q_out),
                                            create_graph=True, 
                                            retain_graph=True, 
                                            allow_unused=True)[0]
            
            # Handle case where gradients are None
            if q_grad is None:
                q_grad = torch.zeros_like(Vii)
            if grad_weights is None:
                grad_weights = torch.zeros_like(elem_weights)
                
            return q_out, q_grad, torch.norm(grad_weights)
        
        # if return_direct:
        #     return q_out
        # elif return_feature:
        #     return (crys_fea, cond_rate, cond_cycle)
        # else:
        #     q_grad = torch.autograd.grad(q_out, Vii, grad_outputs=torch.ones_like(q_out),
        #                         create_graph=True, retain_graph=True)
        #     grad_weights = torch.autograd.grad(q_out, elem_weights, grad_outputs=torch.ones_like(q_out),
        #                                 create_graph=True, retain_graph=True, allow_unused=True)
        #     return q_out, q_grad[0], torch.norm(grad_weights[0])

class build_mlp_batchnorm1d(nn.Module):
    """
    Simple MLP with BatchNorm1d for sequence data.
    """

    def __init__(
        self,
        input_dim=64,
        hidden_dim=32,
        output_dim=32,
        activation=nn.Softplus,
        batchnorm=False,
    ):
        super().__init__()

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = nn.Identity()

        self.act = activation()

    def forward(self, x):
        """
        Forward pass for input shape (batch_size, seq_length, feature_dim).
        """
        # Input shape: (batch_size, seq_length, feature_dim)
        batch_size, seq_length, feature_dim = x.size()

        # Apply the first linear layer
        x = self.hidden(x)  # Shape: (batch_size, seq_length, hidden_dim)

        # Transpose for BatchNorm1d: (batch_size, hidden_dim, seq_length)
        x = x.transpose(1, 2)

        # Apply BatchNorm1d
        x = self.bn(x)

        # Transpose back: (batch_size, seq_length, hidden_dim)
        x = x.transpose(1, 2)

        # Apply activation and final linear layer
        x = self.act(x)
        out = self.fc(x)  # Shape: (batch_size, seq_length, output_dim)

        return out

class CrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim=64, attention_heads=4):
        super(CrossAttentionLayer, self).__init__()

        # 定义一个可以调节的权重 
        self.delta_N = nn.Linear(1, embedding_dim, bias= False)

        # 初始化嵌入层和注意力层
        self.elem_fea_embedding = build_mlp(input_dim= 200+embedding_dim+embedding_dim, output_dim= embedding_dim,
                                         hidden_dim = 2 * embedding_dim,
                                         activation= nn.SiLU, batchnorm= True)

        self.weight_embedding = build_mlp(input_dim= 1, output_dim= embedding_dim,
                                         hidden_dim = 32,
                                         activation= nn.SiLU, batchnorm= True)
        self.cycle_embedding = build_mlp(input_dim= 1, output_dim= embedding_dim,
                                         hidden_dim = 32,
                                         activation= nn.SiLU, batchnorm= True)
        self.vii_embedding = build_mlp(input_dim= 1, output_dim= embedding_dim,
                                         hidden_dim = 32,
                                         activation= nn.SiLU, batchnorm= True)

        # 多头注意力层
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True)

        # Add & Norm
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        # Feed Forward
        self.feed_forward = build_mlp_batchnorm1d(input_dim= embedding_dim, output_dim= embedding_dim,
                                         hidden_dim = 2 * embedding_dim,
                                         activation= nn.SiLU, batchnorm= True)
        
        self.attention_weights = None

    def get_attention_weights(self):
        """返回最近一次的注意力权重"""
        return self.attention_weights
    
    def forward(self, elem_fea, elem_weights, cycle, volatage):
        """
        Modified forward pass to return attention weights
        """
        elem_weights = self.weight_embedding(elem_weights)
        volatage = self.vii_embedding(volatage)
        Q_query = self.elem_fea_embedding(torch.cat([elem_fea, elem_weights, volatage], dim=-1))
        K_query = self.cycle_embedding(cycle)

        q = Q_query.unsqueeze(0)
        k = K_query.unsqueeze(0)
        v = K_query.unsqueeze(0)

        attn_output, attn_weights = self.attn(q, k, v)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 存储注意力权重时保持维度完整性
        self.attention_weights = attn_weights.detach()

        attn_output = attn_output * self.delta_N(cycle - 1)
        attn_output = self.layer_norm1(attn_output + Q_query)
        ff_output = self.feed_forward(attn_output)
        elem_fea_t = self.layer_norm2(ff_output + attn_output)
        elem_fea_t = elem_fea_t.squeeze(0)
        
        return elem_fea_t, self.attention_weights

class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the Roost Model.
    """

    def __init__(
        self,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        weight_pow = 1,
        activation = nn.SiLU,
        batchnorm = False
    ):
        """
        """
        super().__init__()


        self.batchnorm = batchnorm
        self.activation = activation

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        # self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                    weight_pow= weight_pow,
                    activation = self.activation,
                    batchnorm = self.batchnorm,
                )
                for i in range(n_graph)
            ]
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate, activation= self.activation),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg, activation= self.activation,
                                             batchnorm= self.batchnorm),
                    weight_pow = weight_pow,

                )
                for _ in range(cry_heads)
            ]
        )

    def forward(self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx,
                ):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        # elem_fea shape (n, 200)
        elem_fea = self.embedding(elem_fea)  # (n, 31)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)  # (n, 64)/(n,32)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx,
                                  )

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        ## return the head-averaged pooling and the elem_fea_matrix
        return torch.mean(torch.stack(head_fea), dim=0), elem_fea

    def __repr__(self):
        return self.__class__.__name__

class MessageLayer(nn.Module):
    """
    Massage Layers are used to propagate information between nodes in
    the stoichiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg, weight_pow, 
                 activation = nn.LeakyReLU, batchnorm = False):
        """
        """
        super().__init__()

        self.activation = activation
        self.batchnorm = batchnorm

        ######################################不加入边的特征###########################################################
        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * elem_fea_len, 1, elem_gate, activation = self.activation), # +1是为了加入边的特征的维度
                    message_nn=SimpleNetwork(2 * elem_fea_len, elem_fea_len, elem_msg, activation= self.activation,
                                             batchnorm= self.batchnorm),
                    weight_pow = weight_pow,
                )
                for _ in range(elem_heads)
            ]
        )
        ##############################################################################################################

    def forward(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx,
                ):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs

        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # 提取有连接的特征
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]

        # 计算节点与边的连接特征
        # fea = torch.cat([elem_self_fea, elem_nbr_fea, edge_features], dim=1)
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)


        # 使用 attention 聚合
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(
                attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights)
            )

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea

    def __repr__(self):
        return self.__class__.__name__
