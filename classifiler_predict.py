from Encoder import *
import os

ParamList = {
    'ExpName': 'BBBP',
    'MainMetric': 'AUC',
    'DataPath': './datasets/BBBP_SMILESValue.txt',
    'RootPath': './Experiments/',
    'CUDA_VISIBLE_DEVICES': '0',
    'TaskNum': 1,
    'ClassNum': 2,
    'Augmentation': False,
    'Weight': False,

    'ValidRate': 4000,
    'PrintRate': 20,
    'Frag': True,
    'output_size': 1,
    'hidden_state_size':64,
    'atom_feature_size': 39,
    'bond_feature_size': 10,
    'Feature': 'AttentiveFP',

    'ValidBalance': False,
    'TestBalance': False,
    'MaxEpoch': 50,
    'SplitRate': [0.8,0.1],
    'Splitter': 'Random',

    'UpdateRate': 1,
    'LowerThanMaxLimit': 50,
    'DecreasingLimit': 30,

    'FP_size': 150,
    'atom_layers':3,
    'mol_layers':2,
    'DNNLayers':[150],
    'BatchSize':64,
    'drop_rate':0.2,
    'lr':3,
    'WeightDecay':5,

    'SplitValidSeed': 8,
    'SplitTestSeed': 8,
    'TorchSeed': 2
}

os.environ['CUDA_VISIBLE_DEVICES'] = ParamList['CUDA_VISIBLE_DEVICES']

opt = Configs(ParamList)
opt.add_args('SaveDir', opt.args['RootPath'] + opt.args['ExpName'] + '/')
if not os.path.exists(opt.args['SaveDir']):
    os.mkdir(opt.args['SaveDir'])
model_dir = opt.args['SaveDir'] + 'model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

ckpt, value = train_and_evaluate(opt)
