import argparse
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, \
    match_loss, get_time, TensorDataset, epoch

from networks import Generator
import torchvision.models as models


# def main():
parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--lr_gen', type=float, default=0.0002)
parser.add_argument('--lr_model', type=float, default=0.01)

parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--model', type=str, default='ConvNet')
parser.add_argument('--ipc', type=int, default=1)
parser.add_argument('--num_exp', type=int, default=5)
parser.add_argument('--num_eval', type=int, default=1)
parser.add_argument('--Iteration', type=int, default=1)
parser.add_argument('--batch_train', type=int, default=256)
parser.add_argument('--epoch_eval_train', type=int, default=300)
parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
parser.add_argument('--lr_img', type=float, default=0.1)
parser.add_argument('--lr_net', type=float, default=0.01)
parser.add_argument('--data_path', type=str, default='data', help='dataset path')
parser.add_argument('--save_path', type=str, default='result', help='path to save results')

parser.add_argument('--num_worker', type=int, default=0)
parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
args = parser.parse_args()

args.outer_loop, args.inner_loop = get_loops(args.ipc)
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dsa = False

if not os.path.exists(args.data_path):
    os.mkdir(args.data_path)

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

eval_it_pool = np.arange(0, args.Iteration + 1, 500).tolist()
print('eval_it_pool: ', eval_it_pool)

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                     args.data_path,
                                                                                                     args)
args.batch_real = args.ipc

model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

accs_all_exps = dict()
for key in model_eval_pool:
    accs_all_exps[key] = []

data_save = []

print('\n================== Exp 1 ==================\n ')
print('Hyper-parameters: \n', args.__dict__)
print('Evaluation model pool: ', model_eval_pool)

images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
labels_all = [dst_train[i][1] for i in range(len(dst_train))]

images_all = torch.cat(images_all, dim=0).to(args.device)
labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

indices_class = [[] for c in range(num_classes)]
for i, lab in enumerate(labels_all):
    indices_class[lab].append(i)

for c in range(num_classes):
    print('class c = %d: %d real images' % (c, len(indices_class[c])))

def get_images(c, n):
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]


for ch in range(channel):
    print('real images channel %d, mean = %.4f, std = %.4f' % (ch,
                                                               torch.mean(images_all[:, ch]),
                                                               torch.std(images_all[:, ch])))

print('initialize latent vector from random noise')

latent_vector = torch.randn(size=(num_classes * args.ipc, args.latent_dim),
                            dtype=torch.float, device=args.device)
label_syn = torch.tensor(np.array([np.ones(args.ipc) * i for i in range(num_classes)]),
                         dtype=torch.long, device=args.device).view(-1)

generator = Generator(args).to(args.device)
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr_gen, weight_decay=0.2)
optimizer_gen.zero_grad()

criterion = nn.CrossEntropyLoss().to(args.device)
criterion_mse = nn.MSELoss().to(args.device)
print('%s TRAINING BEGINS' % get_time())

for it in range(args.Iteration):
    if it in eval_it_pool:
        for model_eval in model_eval_pool:
            print('------------------------------------------------')
            print('Evaluation')
            print('model_train = %s, model_eval = %s, iteration = %d' % (args.model,
                                                                         model_eval,
                                                                         it))
            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
            print('DC augmentation parameters: \n', args.dc_aug_param)

            accs = []
            for it_eval in range(args.num_eval):
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)

                generator.eval()

                # latent_vector = torch.randn(size=(num_classes * args.ipc, args.latent_dim),
                #                             dtype=torch.float, device=args.device)
                image_syn_eval = generator(latent_vector).detach()
                label_syn_eval = copy.deepcopy(label_syn.detach())

                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval,
                                                         testloader, args)

                accs.append(acc_test)
            print('Evaluate %d random %s, mean = %.4f std = %.4f' % (len(accs),
                                                                     model_eval,
                                                                     np.mean(accs),
                                                                     np.std(accs)))
            print('------------------------------------------------')

            if it == args.Iteration:
                accs_all_exps[model_eval] += accs
