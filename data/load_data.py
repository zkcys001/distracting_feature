import data.preprocess
import torch
import numpy as np
import os
class Dataset(torch.utils.data.Dataset):
    def __init__(self, args,data_files):
        self.data_files = data_files
        self.type_loss=args.type_loss
        self.reg="/"+args.regime
        self.image_type=args.image_type
        self.re = args.regime

    def __getitem__(self, ind):
        data_file = self.data_files[ind]

        #data_file=data_file.replace(self.reg,self.reg+"_s")
        #print("onlyshape")
        data = np.load(data_file)

        x = data[self.image_type]#.reshape(16, 80, 80)
        #x2=np.ones([16,160,160])*255
        #x2[x<255]=0
        #x=np.concatenate([x,x2],0)
        #x = msk_abstract.post_data(x)
        y = data['target']

        style =data_file#[:-4].split("/")[-1].split("_")[3:]

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x = x.type(torch.float32)

        me=data["meta_target"]
        me=torch.from_numpy(me).float()
        return x, y, style,me


    def __len__(self):
        return len(self.data_files)
def load_data(args, data_split):
    data_files = []
    data_dir = '../process_data/reason_data/reason_data/RAVEN-10000/'
    for subdir in os.listdir(data_dir):

        for filename in os.listdir(data_dir + subdir):
            if "npz" in filename:
                data_files.append(data_dir + subdir + "/" + filename)



    df = [data_file for data_file in data_files if data_split in data_file and "npz" in data_file][:]

    #data_files = [data_file for data_file in data_files if data_split in data_file]
    print("Nums of "+data_split+" : ", len(df))
    # train_loader = torch.utils.data.DataLoader(Dataset(train_files), batch_size=args.batch_size, shuffle=True,num_workers=args.numwork)#
    loader = torch.utils.data.DataLoader(Dataset(args,df), batch_size=args.batch_size, num_workers=args.numwork)
    return loader
