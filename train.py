import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import asot

import os
import numpy as np

import utils
import align_dataset
# import align_dataset_multi_action
from models import BaseModel, ConvEmbedder
from config import CONFIG
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import CSVLogger

import torch
import torch.nn as nn
import numpy as np
import argparse

num_eps = 1e-11


class AlignNet(LightningModule):
    def __init__(self, config):
        super(AlignNet, self).__init__()

        self.base_cnn = BaseModel(pretrained=True)

        if config.TRAIN.FREEZE_BASE:
            if config.TRAIN.FREEZE_BN_ONLY:
                utils.freeze_bn_only(module=self.base_cnn)
            else:
                utils.freeze(module=self.base_cnn, train_bn=False)

        self.emb = ConvEmbedder(emb_size=config.DTWALIGNMENT.EMBEDDING_SIZE, l2_normalize=config.LOSSES.L2_NORMALIZE)

        self.lr = config.TRAIN.LR
        self.weight_decay = config.TRAIN.WEIGHT_DECAY
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.freeze_base = config.TRAIN.FREEZE_BASE
        self.freeze_bn_only = config.TRAIN.FREEZE_BN_ONLY

        self.data_path = os.path.abspath(config.DATA_PATH)

        self.hparams.config = config

        self.save_hyperparameters()
        ##########################
        # ASOT Hyperparams:
        self.alpha_train = config.ALPHA_TRAIN
        self.alpha_eval = config.ALPHA_EVAL
        self.n_ot_train = config.N_OT_TRAIN
        self.n_ot_eval = config.N_OT_EVAL
        self.step_size = config.STEP_SIZE
        self.train_eps = config.EPS_TRAIN
        self.eval_eps = config.EPS_EVAL
        self.radius_gw = config.RADIUS_GW
        self.ub_frames = config.UB_FRAMES
        self.ub_actions = config.UB_ACTIONS
        self.lambda_frames_train = config.LAMBDA_FRAMES_TRAIN
        self.lambda_actions_train = config.LAMBDA_ACTIONS_TRAIN
        self.lambda_frames_eval = config.LAMBDA_FRAMES_EVAL
        self.lambda_actions_eval = config.LAMBDA_ACTIONS_EVAL
        self.rho = config.RHO
        self.zeta = config.ZETA
        self.temp = 0.1
        ##########################

    def train(self, mode=True):
        super(AlignNet, self).train(mode=mode)

        if self.freeze_base:
            if self.freeze_bn_only:
                utils.freeze_bn_only(module=self.base_cnn)
            else:
                utils.freeze(module=self.base_cnn, train_bn=False)

    def forward(self, x):
        num_ctxt = self.hparams.config.DATA.NUM_CONTEXT

        num_frames = x.size(1) // num_ctxt
        x = self.base_cnn(x)
        x = self.emb(x, num_frames)
        return x

    def training_step(self, batch, batch_idx):
        # Read AlignData's __getitem__() to understand what is returned in the batch
        # a_X/b_X is a tensor of shape (batchsize=1, 40, Channels=3, Height=224, Width=224) representing the 40 sampled+context frames
        (a_X, _, a_steps, a_seq_len, a_mask), (b_X, _, b_steps, b_seq_len, b_mask) = batch

        # Concatenate the tensors along the batch dimension
        X = torch.cat([a_X, b_X])
        # Pass through the encoder to produce framewise embeddings, the encoder stacks context frames so it outputs less no. of embeddings than the no. of input frames
        embs = self.forward(X)
        # a_embs/b_embs is a tensor of shape (batchsize=1, 20, embeddingsize=128) representing the 20 framewise embeddings
        a_embs, b_embs = torch.split(embs, a_X.size(0), dim=0)

        # Using the variable names used in VAOT
        features_X, features_Y = a_embs, b_embs
        T_X = features_X.shape[1]
        T_Y = features_Y.shape[1]
        mask_X, mask_Y = a_mask, b_mask
        # Eq (6)
        # codes represent a matrix P for each batch element
        # size of a matrix P is (no. of frames in X x no. of frames in Y)
        # P_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y
        codes = torch.exp(features_X @ features_Y.transpose(1, 2) / self.temp)
        codes = codes / codes.sum(dim=-1, keepdim=True)

        # Produce pseudo-labels using ASOT, note that we don't backpropagate through this part
        with torch.no_grad():
            # Calculate the KOT cost matrix from the paragraph above Eq (7)
            # ρR = rho * Temporal prior
            temp_prior = asot.temporal_prior(T_X, T_Y, self.rho, features_X.device)
            # Cost Matrix Ck from section 4.2, no need to divide by norms since both vectors were previously normalized with F.normalize()
            cost_matrix = 1. - features_X @ features_Y.transpose(1, 2)
            # Ĉk = Ck + ρR
            cost_matrix += temp_prior


            ## Added for virtual frames
            B, N, K = cost_matrix.shape
            dev = cost_matrix.device
            top_row = torch.ones(B, 1, K).to(dev) * self.zeta
            cost_matrix = torch.cat((top_row, cost_matrix), dim=1)
            left_column = torch.ones(B, N + 1, 1).to(dev) * self.zeta
            cost_matrix = torch.cat((left_column, cost_matrix), dim=2)


            # opt_codes represent a matrix Tb for each batch element
            # size of a matrix Tb is (no. of frames in X x no. of frames in Y)
            # Tb are the (soft) pseudo-labels defined above Eq (7)
            # Tb_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y

            opt_codes, _ = asot.segment_asot(cost_matrix=cost_matrix, mask_X=mask_X, mask_Y=mask_Y,
                                             eps=self.train_eps, alpha=self.alpha_train, radius=self.radius_gw,
                                             ub_frames=self.ub_frames, ub_actions=self.ub_actions,
                                             lambda_frames=self.lambda_frames_train,
                                             lambda_actions=self.lambda_actions_train,
                                             n_iters=self.n_ot_train, step_size=self.step_size)

        # Eq (7)

        loss_ce = -((opt_codes * torch.log(codes + num_eps)) * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)).sum(dim=2).mean()
        self.log('train_loss', loss_ce)
        return loss_ce

    def configure_optimizers(self):

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        config = self.hparams.config
        train_path = os.path.join(self.data_path, 'train')

        train_transforms = utils.get_transforms(augment=True)
        
        #### This is for training a Multi-Action model on Penn Action 
        # data = align_dataset_multi_action.AlignData(self.data_path, config.TRAIN.NUM_FRAMES, config.DATA, transform=train_transforms, flatten=False)
        
        data = align_dataset.AlignData(train_path, config.TRAIN.NUM_FRAMES, config.DATA, transform=train_transforms, flatten=False)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                        num_workers=config.DATA.WORKERS)

        return data_loader 




def main(hparams):
    seed_everything(hparams.SEED)

    model = AlignNet(hparams)


    try:
        # Check if resuming from a checkpoint
        checkpoint_path = hparams.RESUME_FROM
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Resumed training from checkpoint: {checkpoint_path}")

        checkpoint_callback = utils.CheckpointEveryNSteps(hparams.TRAIN.SAVE_INTERVAL_ITERS, filepath=os.path.join(hparams.CKPT_PATH, 'STEPS'))
        csv_logger = CSVLogger(save_dir=hparams.CKPT_PATH.strip('/').split('/')[-2], name="lightning_logs")

        trainer = Trainer(gpus=[hparams.GPUS], max_epochs=hparams.TRAIN.EPOCHS, default_root_dir=hparams.ROOT,
                          deterministic=True, callbacks=[checkpoint_callback], 
                          limit_val_batches=0, check_val_every_n_epoch=0, num_sanity_val_steps=0,
                          logger=csv_logger, log_every_n_steps=5,
                          resume_from_checkpoint=checkpoint_path)

        trainer.fit(model)

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--ckpt_path', type=str, help='Path to save checkpoints')
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    ###############
    # ASOT args:
    parser.add_argument('--alpha-train', '-at', type=float, default=0.3,
                        help='weighting of KOT term on frame features in OT')
    parser.add_argument('--alpha-eval', '-ae', type=float, default=0.6,
                        help='weighting of KOT term on frame features in OT')
    parser.add_argument('--n-ot-train', '-nt', type=int, nargs='+', default=[25, 1],
                        help='number of outer and inner iterations for ASOT solver (train)')
    parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 1],
                        help='number of outer and inner iterations for ASOT solver (eval)')
    parser.add_argument('--step-size', '-ss', type=float, default=None,
                        help='Step size/learning rate for ASOT solver. Worth setting manually if ub-frames && ub-actions')
    parser.add_argument('--eps-train', '-et', type=float, default=0.07,
                        help='entropy regularization for OT during training')
    parser.add_argument('--eps-eval', '-ee', type=float, default=0.04,
                        help='entropy regularization for OT during val/test')
    parser.add_argument('--radius-gw', '-r', type=float, default=0.02,
                        help='Radius parameter for GW structure loss')
    parser.add_argument('--ub-frames', '-uf', action='store_true',
                        help='relaxes balanced assignment assumption over frames, i.e., each frame is assigned')
    parser.add_argument('--ub-actions', '-ua', action='store_true',
                        help='relaxes balanced assignment assumption over actions, i.e., each action is uniformly represented in a video')
    parser.add_argument('--lambda-frames-train', '-lft', type=float, default=0.05,
                        help='penalty on balanced frames assumption for training')
    parser.add_argument('--lambda-actions-train', '-lat', type=float, default=0.05,
                        help='penalty on balanced actions assumption for training')
    parser.add_argument('--lambda-frames-eval', '-lfe', type=float, default=0.05,
                        help='penalty on balanced frames assumption for test')
    parser.add_argument('--lambda-actions-eval', '-lae', type=float, default=0.01,
                        help='penalty on balanced actions assumption for test')
    parser.add_argument('--rho', type=float, default=0.35,
                        help='Factor for global structure weighting term')
    parser.add_argument('--zeta', type=float, default=0.50,
                        help='If the chance of alignment with all the real frames is less than a certain threshold value, zeta, we align this frame to the virtual frame instead')
    ###############

    args = parser.parse_args()

    CONFIG.GPUS = args.gpus

    if args.root_dir:
        CONFIG.ROOT = args.root_dir
    if args.ckpt_path:
        CONFIG.CKPT_PATH = args.ckpt_path
    if args.data_path:
        CONFIG.DATA_PATH = args.data_path
    if args.num_frames:
        CONFIG.TRAIN.NUM_FRAMES = args.num_frames
        CONFIG.EVAL.NUM_FRAMES = args.num_frames
    if args.workers:
        CONFIG.DATA.WORKERS = args.workers
    CONFIG.RESUME_FROM = args.resume_from
    #################
    # ASOT args stored into config:
    if args.alpha_train is not None:
        CONFIG.ALPHA_TRAIN = args.alpha_train
    if args.alpha_eval:
        CONFIG.ALPHA_EVAL = args.alpha_eval
    if args.n_ot_train:
        CONFIG.N_OT_TRAIN = args.n_ot_train
    if args.n_ot_eval:
        CONFIG.N_OT_EVAL = args.n_ot_eval
    if args.eps_train:
        CONFIG.EPS_TRAIN = args.eps_train
    if args.eps_eval:
        CONFIG.EPS_EVAL = args.eps_eval
    if args.radius_gw is not None:
        CONFIG.RADIUS_GW = args.radius_gw
    if args.lambda_frames_train:
        CONFIG.LAMBDA_FRAMES_TRAIN = args.lambda_frames_train
    if args.lambda_actions_train:
        CONFIG.LAMBDA_ACTIONS_TRAIN = args.lambda_actions_train
    if args.lambda_frames_eval:
        CONFIG.LAMBDA_FRAMES_EVAL = args.lambda_frames_eval
    if args.lambda_actions_eval:
        CONFIG.LAMBDA_ACTIONS_EVAL = args.lambda_actions_eval
    if args.rho is not None:
        CONFIG.RHO = args.rho
    # NOTE: the default values of these args, cause the if condition to fail so we wont use the if condition for them
    CONFIG.ZETA = args.zeta
    CONFIG.STEP_SIZE = args.step_size
    CONFIG.UB_FRAMES = args.ub_frames
    CONFIG.UB_ACTIONS = args.ub_actions
    #################

    main(CONFIG)
