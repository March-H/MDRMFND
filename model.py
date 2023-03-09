import torch
import torch.nn as nn
import numpy as np
from layers import *
from transformers import BertModel

class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, dropout, emb_type ):
        '''
        in the MDFEND paper
        :param emb_dim: 768 in bert or 200 in w2v
        :param mlp_dims: [384]
        :param bert: default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch'
        :param dropout: 0.2
        :param emb_type: default='bert'
        '''
        super(MultiDomainFENDModel, self).__init__()
        self.domain_num = 9 # the number of domains
        self.gamma = 10
        self.num_expert = 5# the number of expert, in the MDFEND, it uses 5 experts
        self.fea_size = 256
        self.emb_type = emb_type
        if (emb_type == 'bert'):
            self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        # build the textCNNs as experts without connection
        # each tectCNN has 768 in_channels and 64 out_channels
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims[-1]),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims[-1], self.num_expert),
                                  nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(multi_head_num=1, input_size=emb_dim,
                                                              output_size=self.fea_size)


    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        if self.emb_type == "bert":
            init_feature = self.bert(inputs, attention_mask=masks)[0]
        elif self.emb_type == 'w2v':
            init_feature = inputs

        feature, _ = self.attention(init_feature, masks)
        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_input_feature = feature
        gate_input = torch.cat([domain_embedding, gate_input_feature], dim=-1)
        gate_value = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))


        return shared_feature