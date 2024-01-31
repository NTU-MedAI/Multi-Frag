import torch as t
import torch.nn as nn
import numpy as np
from Encoder.ClassifierModel import DNN
from Encoder.AttentiveFP import *
from .nnutils_ours import tree_gru_onehot_revised, GatEncoder_raw_gru_revised
import os,pickle,multiprocessing,torch
import dgl
from dgl import DGLGraph
from .mol_tree import Vocab, DGLMolTree
from .chemutils import atom_features, get_mol, get_morgan_fp, bond_features,get_dgl_node_feature,get_dgl_bond_feature
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
'''
class MolPredGraph(nn.Module):
    def __init__(self, feature_size, GCN_layers, DNN_layers, GCNLayertype):
        super(MolPredGraph, self).__init__()
        self.GCN = NaiveGCN(feature_size = feature_size, layers = GCN_layers, GCNLayertype = GCNLayertype)
        self.Classifier = DNN(input_size = 128, layer_sizes = DNN_layers, output_size = 2)
        self.Linear = nn.Linear(GCN_layers[-1], 128)

    def readout(self, FeatureMat):
        MolFeat = t.mean(FeatureMat, dim=-2)
        MolFeat = self.Linear(MolFeat)
        return MolFeat
        #return t.mean(FeatureMat, dim=-2)

    def forward(self, Input):
        [AdjMat, FeatureMat] = Input
        AdjMat = AdjMat.cuda()
        FeatureMat = FeatureMat.cuda()
        X = self.GCN(AdjMat, FeatureMat)           # [batch, atom_num, feature_num]
        X = self.readout(X)                        # [batch, feature_num]
        X = self.Classifier(X)
        return X


class MolPredFP(nn.Module):
    def __init__(self, feature_size, DNN_layers):
        super(MolPredFP, self).__init__()
        self.Classifier = DNN(input_size = feature_size, layer_sizes = DNN_layers, output_size = 2)
    def forward(self, Input):
        X = Input.cuda()
        X = self.Classifier(X)
        return X
'''
class MolPredFragFPv8(nn.Module):
    def __init__(self,
                 atom_feature_size,
                 bond_feature_size,
                 FP_size,
                 atom_layers,
                 mol_layers,
                 DNN_layers,
                 output_size,
                 drop_rate,
                 opt
                 ):
        super(MolPredFragFPv8, self).__init__()
        self.AtomEmbedding = AttentiveFP_atom(
            atom_feature_size = atom_feature_size,
            bond_feature_size = bond_feature_size,
            FP_size = FP_size,
            layers = atom_layers,
            droprate = drop_rate
        )   # For Frags and original mol_graph
        self.MolEmbedding = AttentiveFP_mol(
            layers=mol_layers,
            FP_size=FP_size,
            droprate=drop_rate
        )  # MolEmbedding module can be used repeatedly
        self.Classifier = DNN(
                input_size=856,
                output_size=output_size,
                layer_sizes=DNN_layers,
                opt = opt
        )
        self.AtomEmbeddingHigher = AttentiveFP_atom(
            atom_feature_size = FP_size,
            bond_feature_size = bond_feature_size,
            FP_size = FP_size,
            layers = atom_layers,
            droprate = drop_rate
        )  # For Junction Tree
        #self.InformationFuser =
        self.vocab = Vocab([x.strip("\r\n ") for x in open('/home/ntu/PycharmProjects/Hao/Fra_DulV/vocabulary_molnet.txt')])
        self.hidden_state_size = 128

        self.GATencoder = tree_gru_onehot_revised(vocab=self.vocab, hidden_size=self.hidden_state_size,
                                                  head_nums=4, conv_nums=2)

        self.GATencoder_raw = GatEncoder_raw_gru_revised(hidden_size=self.hidden_state_size,
                                                         head_nums=4, conv_nums=3)

        self.fc1 = nn.Linear(167, 128)
        self.act_func = nn.ReLU()
        # self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, Input,smiles,epoch,key):
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = self.Input_cuda(Input)

        # layer origin
        atom_FP_origin = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list_origin, bond_neighbor_list_origin)
        mol_FP_origin, _ = self.MolEmbedding(atom_FP_origin, atom_mask_origin)

        # layer Frag:
        atom_FP = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list_changed, bond_neighbor_list_changed)
        mol_FP1, activated_mol_FP1 = self.MolEmbedding(atom_FP, frag_mask1)
        mol_FP2, activated_mol_FP2 = self.MolEmbedding(atom_FP, frag_mask2)
        # mol_FP1, mol_FP2 are used to input the DNN module.
        # activated_mol_FP1 and activated_mol_FP2 are used to calculate the mol_FP
        # size: [batch_size, FP_size]
        ##################################################################################
        # Junction Tree Construction
        # construct a higher level graph: Junction Tree

        # Construct atom features of JT:
        batch_size, FP_size = activated_mol_FP1.size()
        pad_node_feature = t.zeros(batch_size, FP_size).cuda()
        JT_atom_features = t.stack([activated_mol_FP1, activated_mol_FP2, pad_node_feature], dim = 1)

        # Junction Tree Construction complete.
        ##################################################################################
        #layer Junction Tree: calculate information of the junction tree of Frags
        atom_FP_super = self.AtomEmbeddingHigher(JT_atom_features,
                                           JT_bond_features,
                                           JT_atom_neighbor_list,
                                           JT_bond_neighbor_list)

        JT_FP, _ = self.MolEmbedding(atom_FP_super, JT_mask)
        raw = []
        tree = []
        wid = []
        fp_list = []
        for smile in smiles:
            raw_graph = self.get_raw_graph(smile)
            raw.append(raw_graph)
            tree_FE, wi = self.get_tree(smile)
            tree.append(tree_FE)
            wid.append(wi)
            MACCS_mol = Chem.MolFromSmiles(smile)
            fp_maccs = AllChem.GetMACCSKeysFingerprint(MACCS_mol)
            fp_list.append(fp_maccs)

        fp_list = torch.Tensor(fp_list)
        fp_list = fp_list.cuda()
        fpn_out = self.fc1(fp_list)
        fpn_out = self.act_func(fpn_out)
        FP = t.cat([mol_FP_origin, mol_FP1,mol_FP2,JT_FP], dim=-1)
        for _wid, mol_tree in zip(wid, tree):  # zip wid,mol_trees,label as tuple
            mol_tree.ndata['wid'] = torch.LongTensor(_wid)
        raw_h, x_r = self.GATencoder_raw(raw,smiles)
        tree = self.test_(tree, raw_h)
        _, Tree_t = self.GATencoder(tree,smiles)
        # entire_FP = t.cat([Tree_t, FP], dim=-1)
        # prediction = self.Classifier(entire_FP)
        # if key==1 or epoch < 19:
        #     pass
        # else:
        #     with open('G1_logits.csv', 'a') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(prediction)
        # return prediction
        if mol_FP_origin.shape[0]==Tree_t.shape[0]:
            entire_FP = t.cat([FP,Tree_t,fpn_out], dim=-1)
            prediction = self.Classifier(entire_FP)
            return prediction
        else:
            Tree_t = Tree_t.repeat(mol_FP_origin.shape[0],1)
            fpn_out = fpn_out.repeat(mol_FP_origin.shape[0],1)
            entire_FP = t.cat([FP,Tree_t,fpn_out], dim=-1)
            prediction = self.Classifier(entire_FP)
            return prediction

    def get_tree(self,smiles):
        vocab_path = '/home/ntu/PycharmProjects/Hao/Fra_DulV/vocabulary_molnet.txt'
        vocab = Vocab([x.strip("\r\n") for x in open(vocab_path)])
        mol_tree = DGLMolTree(smiles)  # mol_tree
        mol_tree = dgl.add_self_loop(mol_tree)  # mol_tree 是否加自环边
        wid = self._set_node_id(mol_tree, vocab)  # idx_cliques
        return mol_tree,wid

    def _set_node_id(self,mol_tree, vocab):  # hash函数，找到mol_tree中每个顶点(官能团簇)在vocab中的索引
        wid = []
        for i, node in enumerate(mol_tree.nodes_dict):
            mol_tree.nodes_dict[node]['idx'] = i
            wid.append(vocab.get_index(mol_tree.nodes_dict[node]['smiles']))
        return wid

    def test_(self, tree, raw_h):
        assert len(tree) == len(raw_h)
        all_data = []
        for i in range(len(raw_h)):
            tt = tree[i].nodes_dict
            r = raw_h[i]
            cliques = []
            for key in tt:
                clique = tt[key]['clique']
                cliques.append(torch.sum(r[clique], dim=0))
            try:
                all_data.append(torch.stack(cliques, dim=0))
            except:
                print(tree[i].smiles)
                all_data.append(torch.sum(r[:], dim=0))
                return
        assert len(all_data) == len(tree)
        for i in range(len(tree)):
            device = torch.device("cpu")
            all_data[i] = all_data[i].to(device)
            tree[i].ndata['h'] = all_data[i]
        return tree

    def get_raw_graph(self,smiles):
        atom_list = []
        mol = get_mol(smiles)
        feats = get_dgl_node_feature(mol)
        mol_raw = DGLGraph()
        mol_raw.add_nodes((len(mol.GetAtoms())))
        mol_raw.ndata['h'] = feats
        for bonds in mol.GetBonds():
            src_id = bonds.GetBeginAtomIdx()
            dst_id = bonds.GetEndAtomIdx()
            mol_raw.add_edges([src_id, dst_id], [dst_id, src_id])
        mol_raw = dgl.add_self_loop(mol_raw)
        e_f = get_dgl_bond_feature(mol)
        mol_raw.edata['e_f'] = e_f
        return mol_raw

    def Input_cuda(self, Input):
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = Input
        if not self.training:
            #print(atom_features.size())
            atom_features = atom_features.squeeze(dim=0).cuda()
            bond_features = bond_features.squeeze(dim=0).cuda()
            atom_neighbor_list_origin = atom_neighbor_list_origin.squeeze(dim=0).cuda()
            bond_neighbor_list_origin = bond_neighbor_list_origin.squeeze(dim=0).cuda()
            atom_mask_origin = atom_mask_origin.squeeze(dim=0).cuda()

            atom_neighbor_list_changed = atom_neighbor_list_changed.squeeze(dim=0).cuda()
            bond_neighbor_list_changed = bond_neighbor_list_changed.squeeze(dim=0).cuda()
            frag_mask1 = frag_mask1.squeeze(dim=0).cuda()
            frag_mask2 = frag_mask2.squeeze(dim=0).cuda()
            bond_index = bond_index.squeeze(dim=0).cuda()

            JT_bond_features = JT_bond_features.squeeze(dim=0).cuda()
            JT_atom_neighbor_list = JT_atom_neighbor_list.squeeze(dim=0).cuda()
            JT_bond_neighbor_list = JT_bond_neighbor_list.squeeze(dim=0).cuda()
            JT_mask = JT_mask.squeeze(dim=0).cuda()

        else:
            atom_features = atom_features.cuda()
            bond_features = bond_features.cuda()
            atom_neighbor_list_origin = atom_neighbor_list_origin.cuda()
            bond_neighbor_list_origin = bond_neighbor_list_origin.cuda()
            atom_mask_origin = atom_mask_origin.cuda()

            atom_neighbor_list_changed = atom_neighbor_list_changed.cuda()
            bond_neighbor_list_changed = bond_neighbor_list_changed.cuda()
            frag_mask1 = frag_mask1.cuda()
            frag_mask2 = frag_mask2.cuda()
            bond_index = bond_index.cuda()

            JT_bond_features = JT_bond_features.cuda()
            JT_atom_neighbor_list = JT_atom_neighbor_list.cuda()
            JT_bond_neighbor_list = JT_bond_neighbor_list.cuda()
            JT_mask = JT_mask.cuda()
        return [atom_features,
                bond_features,
                atom_neighbor_list_origin,
                bond_neighbor_list_origin,
                atom_mask_origin,
                atom_neighbor_list_changed,
                bond_neighbor_list_changed,
                frag_mask1,
                frag_mask2,
                bond_index,
                JT_bond_features,
                JT_atom_neighbor_list,
                JT_bond_neighbor_list,
                JT_mask]

