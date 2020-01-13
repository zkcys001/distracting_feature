import torch
import torch.nn.functional as F

from torch import nn

class Reab3p16(nn.Module):
    def __init__(self,args):
        super(Reab3p16, self).__init__()
        self.NUM_PANELS = 16
        self.type_loss=args.type_loss
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
            nn.Conv2d(16, 32, 3, 2),
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
        self.pre_g_fc = nn.Linear(32 * 9 ** 2, 256)
        self.pre_g_batch_norm = nn.BatchNorm1d(256)

        self.pre_g_fc2 = nn.Linear(32 * 9 ** 2, 256)
        self.pre_g_batch_norm2 = nn.BatchNorm1d(256)

        self.g = nn.Sequential(
            nn.Linear(512+512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256*3),
            nn.BatchNorm1d(256*3),
            nn.ReLU(),
            nn.Linear(256*3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )
        self.g2 = nn.Sequential(
            nn.Linear(512 + 512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 3),
            nn.BatchNorm1d(256 * 3),
            nn.ReLU(),
            nn.Linear(256 * 3, 512),
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
        if self.type_loss:
            self.meta_fc= nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, 9)
            )
    def comp_panel_embedding(self, panel):
        batch_size = panel.shape[0]
        panel = torch.unsqueeze(panel, 1)  # (batch_size, 160, 160) -> (batch_size, 1, 160, 160)
        panel_embedding = self.cnn(panel) # (batch_size, 1, 160, 160) -> (batch_size, 32, 9, 9)
        panel_embedding = panel_embedding.view(batch_size, -1)
        panel_embedding = self.pre_g_fc(panel_embedding)
        panel_embedding = self.pre_g_batch_norm(panel_embedding)
        panel_embedding = F.relu(panel_embedding)
        return panel_embedding

    def panel_comp_obj_pairs(self, objs,batch_size):
        obj_pairses_r =torch.zeros(batch_size, 2, 256*3).cuda()
        obj_pairses_c = torch.zeros(batch_size, 2, 256 * 3).cuda()
        obj_pairses= torch.zeros(batch_size, 54, 256 * 3).cuda()
        count=0
        index=0
        for i in range(8):
            for j in range(i):
                for k in range(j):
                    if ((7-i)==0 and ((7-j)==1) and ((7-k)==2)) or((7-i)==3 and ((7-j)==4) and ((7-k)==5)):

                        obj_pairses_r[:, (7-i)//3, :] = torch.cat(
                            [torch.cat([objs[:, 7 - i, :], objs[:, 7 - j, :]], 1), objs[:, 7 - k, :]], 1)

                        count -= 1
                    elif ((7-i)==0 and ((7-j)==3) and ((7-k)==6)) or((7-i)==1 and ((7-j)==4) and ((7-k)==7)):

                        obj_pairses_c[:, 7-i, :] = torch.cat(
                            [torch.cat([objs[:, 7 - i, :], objs[:, 7 - j, :]], 1), objs[:, 7 - k, :]], 1)
                        count -= 1
                    else:
                        obj_pairses[:,count,:] = torch.cat([torch.cat([objs[:, 7 - i, :], objs[:, 7 - j, :]],1), objs[:, 7-k, :]], 1)
                    #obj_pairs = torch.cat([torch.unsqueeze( objs[:,7-i,:],1),torch.unsqueeze( objs[:,7-j,:],1)],2)
                    #obj_pairses[:,count,:] = torch.cat([obj_pairs, torch.unsqueeze(objs[:, 7-k, :], 1)], 2)
                    count+=1
        return obj_pairses,obj_pairses_c,obj_pairses_r

    def ans_comp_obj_pairs(self, ans,pan,batch_size):
        obj_pairses_r = torch.zeros(batch_size, 1, 256 * 3).cuda()
        obj_pairses_c = torch.zeros(batch_size, 1, 256 * 3).cuda()

        obj_pairses=torch.zeros(batch_size, 26, 256*3).cuda()
        count=0
        for i in range(8):
            for j in range(i):
                if (7-i)==2 and ((7-j)==5)  :
                    obj_pairs = torch.cat([pan[:, 7 - i, :], pan[:, 7 - j, :]], 1)
                    obj_pairses_c[:, 0, :] = torch.cat([obj_pairs, ans], 1)
                    count -= 1
                elif (7-i)==6 and ((7-j)==7) :


                    obj_pairs = torch.cat([pan[:, 7 - i, :], pan[:, 7 - j, :]], 1)
                    obj_pairses_r[:, 0, :] = torch.cat([obj_pairs, ans], 1)
                    count -= 1
                else:
                    obj_pairs = torch.cat([pan[:,7-i,:],pan[:,7-j,:]],1)
                    obj_pairses[:,count,:] = torch.cat([obj_pairs, ans], 1)
                count+=1
        return obj_pairses,obj_pairses_c,obj_pairses_r
    def g_functin(self,context_pairs,panel_embedding_8,num_context_pairs,batch_size):
        context_pairs = torch.cat([context_pairs, panel_embedding_8.repeat(1, num_context_pairs, 1)], 2)
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 1024)
        context_g_out = self.g(context_pairs)
        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)
        context_g_out = context_g_out.sum(1)
        return context_g_out
    def g_functin2(self,context_pairs,panel_embedding_8,num_context_pairs,batch_size):
        context_pairs = torch.cat([context_pairs, panel_embedding_8.repeat(1, num_context_pairs, 1)], 2)
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 1024)
        context_g_out = self.g2(context_pairs)
        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)
        context_g_out = context_g_out.sum(1)
        return context_g_out
    def forward(self, x):
        batch_size = x.shape[0]
        # Compute panel embeddings
        panel_embeddings = torch.zeros(batch_size, self.NUM_PANELS, 256).cuda()
        panel_embedding_8 = self.cnn_global(x[:, :, :, :])
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

        num_context_pairs = 56
        # Compute context pairs once to be used for each answer
        obj_pairses, obj_pairses_c, obj_pairses_r = self.panel_comp_obj_pairs(context_embeddings,batch_size)# (batch_size, 56, 256*3)

        '''context_pairs = torch.cat([context_pairs, panel_embedding_8.repeat(1, num_context_pairs, 1)], 2)
        context_pairs = context_pairs.view(batch_size * num_context_pairs, 1024)
        context_g_out = self.g(context_pairs)
        context_g_out = context_g_out.view(batch_size, num_context_pairs, 512)'''
        context_g_out1= self.g_functin2(obj_pairses,panel_embedding_8,54,batch_size)
        context_g_outr = self.g_functin(obj_pairses_r, panel_embedding_8, 2, batch_size)
        context_g_outc = self.g_functin(obj_pairses_c, panel_embedding_8, 2, batch_size)
        context_g_out = context_g_out1+context_g_outc+context_g_outr
        f_out = torch.zeros(batch_size, int(self.NUM_PANELS/2)).cuda()

        if self.type_loss:f_meta=torch.zeros(batch_size, 512).cuda()

        for answer_ind in range(8):
            answer_embedding = answer_embeddings[:, answer_ind, :] # (batch_size, 256)

            context_answer_pairs,context_answer_pairs_c,context_answer_pairs_r = self.ans_comp_obj_pairs(answer_embedding,context_embeddings,batch_size)# (batch_size, 28, 512)

            '''context_answer_pairs = torch.cat([context_answer_pairs, panel_embedding_8.repeat(1, 28, 1)], 2)
            context_answer_pairs = context_answer_pairs.view(batch_size * 28, 1024)
            context_answer_g_out = self.g(context_answer_pairs) # (8, 512)
            context_answer_g_out = context_answer_g_out.view(batch_size, 28, 512)
            context_answer_g_out = context_answer_g_out.sum(1)'''
            context_answer_g_out1 = self.g_functin2(context_answer_pairs, panel_embedding_8, 26, batch_size)
            context_answer_g_outr = self.g_functin(context_answer_pairs_r, panel_embedding_8, 1, batch_size)
            context_answer_g_outc = self.g_functin(context_answer_pairs_c, panel_embedding_8, 1, batch_size)
            context_answer_g_out = context_answer_g_out1 + context_answer_g_outc  + context_answer_g_outr
            g_out = context_g_out + context_answer_g_out
            if self.type_loss:f_meta+=g_out
            f_out[:, answer_ind] = self.f(g_out).squeeze()

        if self.type_loss:
            return F.log_softmax(f_out, dim=1),F.sigmoid(self.meta_fc(f_meta))
        else:
            return F.log_softmax(f_out, dim=1)