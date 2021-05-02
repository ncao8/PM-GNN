import os.path as osp
import json
import collections
import numpy as np 
import torch
import numpy as np
import scipy.sparse as sp
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch_geometric.data import InMemoryDataset, Data
import os

class Autopo(InMemoryDataset):
    r"""The Flickr dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing descriptions and common properties of images.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """


    def __init__(self, root, transform=None, pre_transform=None, data_path_root=None):
        self.data_path_root = '.'
        super(Autopo, self).__init__(root, transform, pre_transform)
        # tmp = self.get_tmp()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.json']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # process sample
    def get_tmp(self):

        print("get_tmp running")

        json_file = json.load(open(self.data_path_root + "/dataset.json"))

        tmp = {}
       # first run of all data
        for item in json_file:
            # count += 1
            file_name = item

            list_of_edge = json_file[item]["list_of_edge"]
            list_of_node = json_file[item]["list_of_node"]

            edge_attr = json_file[item]["edge_attr"]
            node_attr = json_file[item]["node_attr"]

#            tmp_vout=[0,0,0]
#            if (abs(json_file[item]["vout"]/100)>1.2):
#                  tmp_vout=[1,0,0]
#            elif (abs(json_file[item]["vout"]/100)<0.7):
#                  tmp_vout=[0,1,0]
#            else:
#                tmp_vout=[0,0,1]
#
            tmp_vout=0
            if (abs(json_file[item]["vout"]/100)>1.5):
                  tmp_vout=0
            elif (abs(json_file[item]["vout"]/100)<0.5):
                  tmp_vout=1
            else:
                tmp_vout=0


            target_vout=[]
#            target_vout.append(json_file[item]["vout"]/100)
#            target_vout.append(json_file[item]["vout"]/100)
            target_vout.append(tmp_vout)
            target_eff=[]
            target_eff.append(json_file[item]["eff"])
#            if json_file[item]["eff"]>0.7:
#                target_eff.append(1)
#            else:
#                target_eff.append(0)
             
            rout=json_file[item]["rout"]
            cout=json_file[item]["cout"]
            freq=json_file[item]["freq"]
            duty_cycle=json_file[item]["duty_cycle"]

            if json_file[item]["vout"]==-1:
                continue

            if json_file[item]["eff"]<0 or json_file[item]["eff"]>1:
                continue

            tmp_list_of_edge = []

            print(json_file[item]["netlist"])

            list_of_node_name = []
            list_of_edge_name = []
 
            for node in node_attr:
                list_of_node_name.append(node)

            for edge in edge_attr:
                print(edge)
                list_of_edge_name.append(edge)

            for edge in list_of_edge:
                tmp_list_of_edge.append(edge[:])

            node_to_delete=[]
            node_to_replace=[]

            for edge in tmp_list_of_edge:
                if edge[0] not in list_of_edge_name:
                    node_to_delete.append(str(edge[1]))
                    node_to_replace.append(edge[0])

            list_of_edge_new=[]
            for edge in tmp_list_of_edge:
                print(edge)
                if str(edge[1]) in node_to_delete:
                    index=node_to_delete.index(str(edge[1]))
                    edge[1]=node_to_replace[index]
                if edge[0] not in node_to_replace:
                    list_of_edge_new.append(edge[:])

            list_of_node_new=[]

            for node in list_of_node_name:
                if node not in node_to_delete:
                    list_of_node_new.append(node)


            node_attr_new=[]
            for node in list_of_node_new:
                node_attr_new.append(node_attr[node][:])
            
            edge_start=[]
            edge_end=[]

            edge_attr_new1=[]
            edge_attr_new2=[]

            for e1 in edge_attr:
                counter=0
                start=-1
                end=-1
                for e2 in list_of_edge_new:
                    if e2[0]==e1 and counter==0:
                        counter=counter+1
                        start=list_of_node_new.index(str(e2[1]))
                    if e2[0]==e1 and counter==1:
                        end=list_of_node_new.index(str(e2[1]))
                if start==-1 or end==-1:
                    print("ERROR in finding edge index")
                    continue
                edge_start.append(start)
                edge_start.append(end)
                edge_end.append(end)
                edge_end.append(start)
                if e1[0]!='S':
                    edge_attr_new1.append(edge_attr[e1])
                    edge_attr_new1.append(edge_attr[e1])
                    edge_attr_new2.append(edge_attr[e1])
                    edge_attr_new2.append(edge_attr[e1])
                else:
                    edge_attr_new1.append(edge_attr[e1])
                    edge_attr_new1.append(edge_attr[e1])
                    tmp_name=''
                    if e1[:2]=='Sa':
                        tmp_name='Sb0'
                    else:
                        tmp_name='Sa0'
                    edge_attr_new2.append(edge_attr[tmp_name])
                    edge_attr_new2.append(edge_attr[tmp_name])
                    

            edge_attr_new1.append([0,1/cout,0])
            edge_attr_new1.append([0,1/cout,0])
            edge_attr_new2.append([0,1/cout,0])
            edge_attr_new2.append([0,1/cout,0])
            edge_attr_new1.append([1/rout,0,0])
            edge_attr_new1.append([1/rout,0,0])
            edge_attr_new2.append([1/rout,0,0])
            edge_attr_new2.append([1/rout,0,0])
 


#            print(edge_attr_new1)
#            print(edge_attr_new2)

            VOUT_index=list_of_node_new.index('VOUT')
            GND_index=list_of_node_new.index('GND')
            VIN_index=list_of_node_new.index('VIN')

            edge_start.append(VOUT_index)
            edge_start.append(GND_index)
            edge_end.append(GND_index)
            edge_end.append(VOUT_index)
            edge_start.append(VOUT_index)
            edge_start.append(GND_index)
            edge_end.append(GND_index)
            edge_end.append(VOUT_index)
 
            edge_index=[]
            edge_index=[edge_start, edge_end]

            tmp[file_name]={}

            tmp[file_name]['nodes'] = node_attr_new
            tmp[file_name]['edge_index'] = edge_index
            tmp[file_name]['edge_attr1'] = edge_attr_new1
            tmp[file_name]['edge_attr2'] = edge_attr_new2 
            tmp[file_name]['target_vout'] = target_vout
            tmp[file_name]['target_eff'] = target_eff

#            print("\n")
            print(file_name)
            print(json_file[file_name]['netlist'])
            print(tmp[file_name]['nodes'])
            print(tmp[file_name]['edge_index'])
            print(tmp[file_name]['edge_attr1'])
            print(tmp[file_name]['edge_attr2'])

            print(target_vout,target_eff)



        return tmp

    def process(self):

        print("process running")

        count = 0
       
        tmp = self.get_tmp()
        data_list=[]
        for fn in tmp:
            x=torch.tensor(tmp[fn]["nodes"],dtype=torch.float)
            edge_index=torch.tensor(tmp[fn]["edge_index"],dtype=torch.long)
            edge_attr1=torch.tensor(tmp[fn]["edge_attr1"],dtype=torch.float)
            edge_attr2=torch.tensor(tmp[fn]["edge_attr2"],dtype=torch.float)
            target_vout=torch.tensor(tmp[fn]["target_vout"],dtype=torch.float)
            target_eff=torch.tensor(tmp[fn]["target_eff"],dtype=torch.float)

            data=Data(x=x,edge_index=edge_index,edge_attr1=edge_attr1,edge_attr2=edge_attr2,y=target_vout,y_eff=target_eff)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

       
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
