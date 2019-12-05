import os
import random

import numpy as np
import torch.utils.data

import misc
from data.structure import RelationType, ObjectType, AttributeType, Triple, Structure


class PreprocessDataset(torch.utils.data.Dataset):
    '''This is only used to load data for preprocessing purposes.'''
    def __init__(self, data_files, key):
        self.data_files = data_files
        self.key = key

    def __getitem__(self, ind):
        data_file = self.data_files[ind]
        data = np.load(data_file)
        return data[self.key]

    def __len__(self):
        return len(self.data_files)

def parse_structure(structure_arr):
    structure = Structure()
    for triple_arr in structure_arr:
        triple_arr = list(triple_arr)
        triple_arr = [elem.decode('UTF-8') for elem in triple_arr]
        triple_arr = [elem.upper() for elem in triple_arr]
        triple = Triple(RelationType[triple_arr[2]], ObjectType[triple_arr[0]], AttributeType[triple_arr[1]])
        structure.add(triple)
    return structure

def save_structure_to_files(regime):
    '''Parse the metadata to compute a mapping between structures and their corresponding datapoints.'''
    print('Saving structure metadata')
    data_dir = '../data/'+regime+'/'
    data_files = os.listdir(data_dir)
    data_files = [data_dir + data_file for data_file in data_files]
    structure_to_files = {}
    for count,data_file in enumerate(data_files):
        if count % 10000 == 0:
            print(count, '/', len(data_files))
        data = np.load(data_file)
        # TODO Serialize structure instead of using string representation
        structure = parse_structure(data['relation_structure']).to_str()
        if structure not in structure_to_files:
            structure_to_files[structure] = [data_file]
        else:
            structure_to_files[structure].append(data_file)
    if os.path.exists('save_state/'+regime):
        os.mkdir('save_state/'+regime)
    misc.save_file(structure_to_files, 'save_state/'+regime+'/structure_to_files.pkl')
    return structure_to_files

def compute_data_files(regime, n, args):
    '''Sort the data files in increasing order of complexity, then return the n least complex datapoints.'''
    data_dir = '../data/'+regime+'/'
    if n == -1:
        data_files = os.listdir(data_dir)
        data_files = [data_dir + data_file for data_file in data_files]
        return data_files
    elif n == -2:
        test_files = [data_dir + data_file for data_file in os.listdir(data_dir) if 'test' in data_file]
        data_files = []
        if os.path.exists('save_state/' + regime + '/structure_to_files.pkl'):
            print('Loading structure metadata')
            structure_to_files = misc.load_file('save_state/' + regime + '/structure_to_files.pkl')
        else:
            structure_to_files = save_structure_to_files(regime)
        all_structure_strs = list(structure_to_files.keys())
        all_structure_strs.sort(key=lambda x: x.count(','))
        for structure_str in all_structure_strs:
            data_i=structure_to_files[structure_str]
            if len(data_i)>5000:
                data_i=data_i[:5000]
            data_files.extend(data_i)
            #print(structure_str, len(structure_to_files[structure_str]))
        data_files=[data_file for data_file in data_files if 'train' in data_file]
        data_files.extend(test_files)
        return data_files
    elif n == -3:

        data_files = []
        if os.path.exists('save_state/' + regime + '/structure_to_files.pkl'):
            print('Loading structure metadata')
            structure_to_files = misc.load_file('save_state/' + regime + '/structure_to_files.pkl')
        else:
            structure_to_files = save_structure_to_files(regime)
        all_structure_strs = list(structure_to_files.keys())
        all_structure_strs.sort(key=lambda x: x.count(','))
        for structure_str in all_structure_strs:
            data_i=structure_to_files[structure_str]
            if len(data_i)>20000 or len(data_i)<10000:
                continue
            data_files.extend(data_i)
            print(structure_str, len(structure_to_files[structure_str]))

        return data_files
    elif n==-4:
        data_files = os.listdir("/home/zkc/reason/andshapecolormask/")
        data_files = ["/home/zkc/reason/andshapecolormask/" + data_file for data_file in data_files]
        return data_files

    else:
        data_files = []
        if os.path.exists('save_state/'+regime+'/structure_to_files.pkl'):
            print('Loading structure metadata')
            structure_to_files = misc.load_file('save_state/'+regime+'/structure_to_files.pkl')
        else:
            structure_to_files = save_structure_to_files(regime)
        all_structure_strs = list(structure_to_files.keys())
        # The number of commas in the structure_str is used as a proxy for complexity

        i=0
        all_structure_strs.sort(key=lambda x: x.count(','))


        for structure_str in all_structure_strs:

            data_i=structure_to_files[structure_str]
            if args.image_type=="image":
                if "SHAPE" in structure_str and "),("not in structure_str:
                    data_files.extend(data_i)
                    print(structure_str, ":", len(data_i))
                if "LINE" in structure_str and "),("not in structure_str:
                    data_files.extend(data_i)
                    print(structure_str, ":", len(data_i))
            elif args.image_type=="shape_im":
                if "SHAPE" in structure_str and "),("not in structure_str:
                    data_files.extend(data_i)
                    print(structure_str, ":", len(data_i))
            elif args.image_type=="line_im":
                if "LINE" in structure_str and "),("not in structure_str:
                    data_files.extend(data_i)
                    print(structure_str, ":", len(data_i))




        return data_files

'''
class RelationType(Enum):
    PROGRESSION = 1
    XOR = 2
    OR = 3
    AND = 4
    CONSISTENT_UNION = 5
class ObjectType(Enum):
    SHAPE = 1
    LINE = 2
class AttributeType(Enum):
    SIZE = 1
    TYPE = 2
    COLOR = 3
    POSITION = 4
    NUMBER = 5
'''
def provide_data(regime, n,a,s):
    #code = ['SHAPE', 'LINE', "COLOR", 'NUMBER', 'POSITION', 'SIZE','TYPE',
    #        'PROGRESSION', "XOR", "OR", 'AND', 'CONSISTENT_UNION']
    base=4000
    data_files=[]
    data_dir = '/home/lab/zkc/reason/process_data/reason_data/reason_data/RAVEN-10000/'
    for subdir in os.listdir(data_dir):

        for filename in os.listdir(data_dir + subdir):
            if "npz" in filename and "train" in filename:
                data_files.append(data_dir+subdir+"/"+filename)


    train_files=[[] for _ in range(n)]
    for data_file in data_files:
        name_=data_file[:-4].split("/")[-1].split("_")[3:]
        for number_ in name_:
            train_files[int(number_)].append(data_file)


    df=[]
    for i in range(n):

        random.shuffle(train_files[i])
        df.extend(train_files[i][:int(base*a[i])])
        #print(x, int(base*a[i]))

    return df



def save_normalization_stats(regime, batch_size=100):
    '''Compute the mean and standard deviation jointly across all channels.'''
    print('Saving normalization stats')
    data_dir = '../data/'+regime+'/'
    data_files = os.listdir(data_dir)
    data_files = [data_dir + data_file for data_file in data_files]
    train_files = [data_file for data_file in data_files if 'train' in data_file]
    loader = torch.utils.data.DataLoader(PreprocessDataset(train_files, 'image'), batch_size=batch_size)
    print('Computing x_mean')
    sum = 0
    n = 0
    count = 0
    for x in loader:
        sum += x.sum()
        n += x.numel()
        count += batch_size
        if count % 100000 == 0:
            print(count, '/', len(train_files))
    x_mean = float(sum / n)
    print('Computing x_sd')
    sum = 0
    n = 0
    count = 0
    for x in loader:
        sum += ((x - x_mean)**2).sum()
        n += x.numel()
        count += batch_size
        if count % 100000 == 0:
            print(count, '/', len(train_files))
    x_sd = float(np.sqrt(sum / n))
    misc.save_file((x_mean, x_sd), 'save_state/'+regime+'/normalization_stats.pkl')
    return x_mean, x_sd