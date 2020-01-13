import torch
import torch.nn.functional as F

from torch import nn

class WildRelationNet(nn.Module):
    def __init__(self):
        super(WildRelationNet, self).__init__()
        self.NUM_PANELS = 16

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.cnn_global = nn.Sequential(
            nn.Conv2d(8, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pre_g_fc = nn.Linear(32 * 4 ** 2, 256)
        self.pre_g_batch_norm = nn.BatchNorm1d(256)

        self.pre_g_fc2 = nn.Linear(32 * 4 ** 2, 256)
        self.pre_g_batch_norm2 = nn.BatchNorm1d(256)

        self.g = nn.Sequential(
            nn.Linear(512+256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.f = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 1)
        )

    def comp_panel_embedding(self, panel):
        batch_size = panel.shape[0]
        panel = torch.unsqueeze(panel, 1)  # (batch_size, 160, 160) -> (batch_size, 1, 160, 160)
        panel_embedding = self.cnn(panel) # (batch_size, 1, 160, 160) -> (batch_size, 32, 9, 9)
        panel_embedding = panel_embedding.view(batch_size,-1)
        panel_embedding = self.pre_g_fc(panel_embedding)
        panel_embedding = self.pre_g_batch_norm(panel_embedding)
        panel_embedding = F.relu(panel_embedding)
        return panel_embedding

    def comp_obj_pairs(self, objs):
        """
        The row-wise pairs of

        [[1, 1],
         [2, 2],
         [3, 3]]

        are formed by horizontally concatenating

        [[1, 1],     [[1, 1],     [[1, 1, 1, 1],
         [2, 2],      [1, 1],      [2, 2, 1, 1],
         [3, 3]],     [1, 1],      [3, 3, 1, 1],
        [[1, 1],     [[2, 2],     [[1, 1, 2, 2],
         [2, 2],  +   [2, 2],  =   [2, 2, 2, 2],
         [3, 3]],     [2, 2],      [3, 3, 2, 2],
        [[1, 1],     [[3, 3],     [[1, 1, 3, 3],
         [2, 2],      [3, 3],      [2, 2, 3, 3],
         [3, 3]]      [3, 3]]      [3, 3, 3, 3]]
        """
        num_objs = objs.shape[1]
        obj_lhs = torch.unsqueeze(objs, 1) # (batch_size, 8, 256) -> (1, 1, 8, 256)
        obj_lhs = obj_lhs.repeat(1, num_objs, 1, 1) # (batch_size, 1, 8, 256) -> (1, 8, 8, 256)
        obj_rhs = torch.unsqueeze(objs, 2) # (batch_size, 8, 256) -> (1, 8, 1, 256)
        obj_rhs = obj_rhs.repeat(1, 1, num_objs, 1) # (batch_size, 8, 1, 256) -> (1, 8, 8, 256)
        obj_pairs = torch.cat([obj_lhs, obj_rhs], 3)
        return obj_pairs

    def forward(self, x):
        batch_size = x.shape[0]
        # Compute panel embeddings
        panel_embeddings = torch.zeros(batch_size, self.NUM_PANELS, 256).cuda()
        panel_embedding_8 = self.cnn_global(x[:, 0:8, :, :])
        panel_embedding_8 = self.pre_g_fc2(panel_embedding_8.view(batch_size, -1))
        panel_embedding_8 = self.pre_g_batch_norm2(panel_embedding_8)
        panel_embedding_8 = F.relu(panel_embedding_8)
        panel_embedding_8 = torch.unsqueeze(panel_embedding_8, 1)
        for panel_ind in range(self.NUM_PANELS):
            panel = x[:, panel_ind, :, :]
            panel_embedding = self.comp_panel_embedding(panel)
            panel_embeddings[:, panel_ind, :] = panel_embedding
        context_embeddings = panel_embeddings[:, :int(self.NUM_PANELS/2), :] # (batch_size, 8, 256)
        answer_embeddings = panel_embeddings[:, int(self.NUM_PANELS/2):, :] # (batch_size, 8, 256)
        # Compute context pairs once to be used for each answer
        context_pairs = self.comp_obj_pairs(context_embeddings)# (batch_size, 8,8, 512)
        #print(panel_embedding_8.shape,context_pairs.shape)
        context_pairs = torch.cat([context_pairs,torch.unsqueeze( panel_embedding_8,2).repeat(1, 8,8, 1)], 3)
        num_context_pairs = 64
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 512+256)
        context_g_out = self.g(context_pairs)

        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)
        context_g_out = context_g_out.sum(1)
        f_out = torch.zeros(batch_size, int(self.NUM_PANELS/2)).cuda()
        for answer_ind in range(8):
            answer_embedding = answer_embeddings[:, answer_ind, :] # (batch_size, 256)
            answer_embedding = torch.unsqueeze(answer_embedding, 1) # (batch_size, 1, 256)
            answer_embedding_pairs =  torch.cat([answer_embedding, answer_embedding], 2)
            answer_embedding = answer_embedding.repeat(1, 8, 1) # (batch_size, 1, 256) -> (batch_size, 8, 256)
            # Compute pairs by horizontal concatenation (same idea as context-context pairs)
            context_answer_pairs = torch.cat([context_embeddings, answer_embedding], 2) # (batch_size, 8, 512)
            context_answer_pairs = torch.cat([context_answer_pairs, answer_embedding_pairs], 1)# (batch_size, 9, 512)
            context_answer_pairs = torch.cat([context_answer_pairs, panel_embedding_8.repeat(1, 9, 1)], 2)
            context_answer_pairs = context_answer_pairs.view(batch_size * 9, 512+256)
            context_answer_g_out = self.g(context_answer_pairs) # (8, 512)
            context_answer_g_out = context_answer_g_out.view(batch_size, 9, 512)
            context_answer_g_out = context_answer_g_out.sum(1)
            g_out = context_g_out + context_answer_g_out
            f_out[:, answer_ind] = self.f(g_out).squeeze()

        ff=F.log_softmax(f_out, dim=1)
        return ff,ff