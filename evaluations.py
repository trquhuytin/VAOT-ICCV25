import os
import numpy as np
import csv

import utils
import align_dataset_test as align_dataset
from config import CONFIG

from evals.phase_classification import evaluate_phase_classification, compute_ap
from evals.kendalls_tau import evaluate_kendalls_tau
from evals.phase_progression import evaluate_phase_progression

from train import AlignNet
import torch
from torch.utils.tensorboard import SummaryWriter

import random
import argparse
import glob
from natsort import natsorted

def get_embeddings(model, data, labels_npy, args):

    embeddings = []
    labels = []
    frame_paths = []
    names = []

    device = f"cuda:{args.device}"

    for act_iter in iter(data):

        for i, seq_iter in enumerate(act_iter):
            seq_embs = []
            seq_fpaths = []
            original = 0
            for _, batch in enumerate(seq_iter):
                a_X, a_name, a_frames = batch
                a_X = a_X.to(device).unsqueeze(0)
                original = a_X.shape[1]//2
                
                b =  a_X[:, -1].clone()
                try:
                    b = torch.stack([b]*((args.num_frames*2)-a_X.shape[1]),axis=1).to(device)
                except:
                    b = torch.from_numpy(np.array([])).float().to(device)
                a_X = torch.concat([a_X,b], axis=1)
                a_emb = model(a_X)[:, :original,:]

                if args.verbose:
                    print(f'Seq: {i}, ', a_emb.shape)

                seq_embs.append(a_emb.squeeze(0).detach().cpu().numpy())
                seq_fpaths.extend(a_frames)
            
            seq_embs = np.concatenate(seq_embs, axis=0)
            
            name = str(a_name).split('/')[-1]

            lab = labels_npy[name]['labels']

            # This deals with the issue when there's a length mismatch in labels and frames
            end = min(seq_embs.shape[0], len(lab))
            lab = lab[:end]
            seq_embs = seq_embs[:end]
            # make the no. of framepaths same as the no. of embs 
            seq_fpaths = seq_fpaths[:end]
            print(": : : Running get_embeddings() : : :")
            embeddings.append(seq_embs[:end])
            frame_paths.append(seq_fpaths)
            names.append(a_name)
            labels.append(lab)

    return embeddings, names, labels, frame_paths
         

def main(ckpts, args):
    
    # summary_dest = os.path.join(args.dest, 'eval_logs')
    # os.makedirs(summary_dest, exist_ok=True)

    # Make the log directory and Define the CSV file path
    os.makedirs(args.dest, exist_ok=True)
    csv_path = os.path.join(args.dest, 'evaluation_results.csv')
    
    # Default headers for most datasets, comment out for IKEA ASM
    csv_headers = ['step', 'PC (0.1)', 'PC (0.5)', 'PC (1.0)', 
                   'Progression', 'Kendall Tau', 'AP@5', 'AP@10', 'AP@15']
    
    # Use these headers for IKEA ASM
    # csv_headers = ['step', 'PC (0.1)', 'PC (0.5)', 'PC (1.0)', 'AP@5', 'AP@10', 'AP@15']

    # Create the csv file and write the header
    with open(csv_path, 'w', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(csv_headers)
    
    for ckpt in ckpts:
        # Init the dict to store the next row of the csv
        csv_dict = dict()

        writer = None
        # writer = SummaryWriter(summary_dest, filename_suffix='eval_logs')
        
        print(f"\n\nStarting Evaluation On Checkpoint: {ckpt}\n\n")

        # get ckpt-step from the ckpt name
        _, ckpt_step = ckpt.split('.')[0].split('_')[-2:]
        ckpt_step = int(ckpt_step.split('=')[1])
        DEST = os.path.join(args.dest, 'eval_step_{}'.format(ckpt_step))

        device = f"cuda:{args.device}"
        model = AlignNet.load_from_checkpoint(ckpt, map_location=device)
        model.to(device)
        model.eval()

        # grad off
        torch.set_grad_enabled(False)
        
        if args.num_frames:
            CONFIG.TRAIN.NUM_FRAMES = args.num_frames
            CONFIG.EVAL.NUM_FRAMES = args.num_frames
        
        CONFIG.update(model.hparams.config)
        
        if args.data_path:
            data_path = args.data_path
        else:
            data_path = CONFIG.DATA_PATH
        data_path = './Data_Test/'
        
        train_path = os.path.join(data_path, 'Test')
        val_path = os.path.join(data_path, 'Test')
        lab_name = "pouring" + "_val"
        labels = np.load(f"./npyrecords/{lab_name}.npy", allow_pickle=True).item()

        # create dataset
        _transforms = utils.get_transforms(augment=False)

        random.seed(0)
        train_data = align_dataset.AlignData(train_path, args.batch_size, CONFIG.DATA, transform=_transforms, flatten=False)
        val_data = align_dataset.AlignData(val_path, args.batch_size, CONFIG.DATA, transform=_transforms, flatten=False)
        

        all_classifications = []
        all_kendalls_taus = []
        all_phase_progressions = []
        ap5, ap10, ap15 = 0, 0, 0

        for i_action in range(train_data.n_classes):

            train_data.set_action_seq(i_action)
            val_data.set_action_seq(i_action)

            train_act_name = train_data.get_action_name(i_action)
            val_act_name = val_data.get_action_name(i_action)
            
            assert train_act_name == val_act_name
            
            if args.verbose:
                print(f'Getting embeddings for {train_act_name}...')

            val_embs, val_names, val_labels, val_frame_paths = get_embeddings(model, val_data, labels, args)
            # train and val are the exact same data now
            train_embs, train_names, train_labels, train_frame_paths = val_embs, val_names, val_labels, val_frame_paths

#### Commented out this block(saving & loading embeddings) to save time on evaluation
            # # save embeddings
            # os.makedirs(DEST, exist_ok=True)
            # DEST_TRAIN = os.path.join(DEST, f'train_{train_act_name}_embs.npy')
            # DEST_VAL = os.path.join(DEST, f'val_{val_act_name}_embs.npy')

            # np.save(DEST_TRAIN, {'embs' : train_embs, 'names':train_names, 'labels': train_labels, 'frame_paths': train_frame_paths})
            # np.save(DEST_VAL,   {'embs' : val_embs,   'names':val_names,   'labels': val_labels,   'frame_paths': val_frame_paths})
            
            # train_embeddings = np.load(DEST_TRAIN, allow_pickle=True).tolist()
            # val_embeddings = np.load(DEST_VAL, allow_pickle=True).tolist()

            # train_embs, train_labels, train_names, train_frame_paths = train_embeddings['embs'], train_embeddings['labels'], train_embeddings['names'], train_embeddings['frame_paths']
            # val_embs, val_labels, val_names, val_frame_paths = val_embeddings['embs'], val_embeddings['labels'], val_embeddings['names'], val_embeddings['frame_paths']
#####################################################################################

            # Evaluating Classification
            train_acc, val_acc = evaluate_phase_classification(ckpt_step, train_embs, train_labels, val_embs, val_labels, 
                                                                act_name=train_act_name, CONFIG=CONFIG, writer=writer, verbose=args.verbose, csv_dict=csv_dict)
            
            # Note: if you want to log the frame retrievals and scores, then pass log=True to compute_ap() 
            ap5, ap10, ap15 = compute_ap(val_embs, val_labels, val_names, val_frame_paths)

            all_classifications.append([train_acc, val_acc])


#### Comment out this block for IKEA ASM
            # Evaluating Kendall's Tau
            train_tau, val_tau = evaluate_kendalls_tau(train_embs, val_embs, stride=args.stride,
                                                        kt_dist=CONFIG.EVAL.KENDALLS_TAU_DISTANCE, visualize=False)
            all_kendalls_taus.append([train_tau, val_tau])
            
            if writer:
                writer.add_scalar(f'kendalls_tau/train_{train_act_name}', train_tau, global_step=ckpt_step)
                writer.add_scalar(f'kendalls_tau/val_{val_act_name}', val_tau, global_step=ckpt_step)

            # Evaluating Phase Progression
            _train_dict = {'embs': train_embs, 'labels': train_labels}
            _val_dict = {'embs': val_embs, 'labels': val_labels}
            train_phase_scores, val_phase_scores = evaluate_phase_progression(_train_dict, _val_dict, "_".join(lab_name.split('_')[:-1]),
                                                                                ckpt_step, CONFIG, writer=writer, verbose=args.verbose)

            all_phase_progressions.append([train_phase_scores[-1], val_phase_scores[-1]])
        
        train_kendalls_tau, val_kendalls_tau = np.mean(all_kendalls_taus, axis=0)
        train_phase_prog, val_phase_prog = np.mean(all_phase_progressions, axis=0)
########################################

        train_classification, val_classification = np.mean(all_classifications, axis=0)

        if writer:
            writer.add_scalar('metrics/AP@5_val', ap5, global_step=ckpt_step)
            writer.add_scalar('metrics/AP@10_val', ap10, global_step=ckpt_step)
            writer.add_scalar('metrics/AP@15_val', ap15, global_step=ckpt_step)
            
            writer.add_scalar('metrics/all_classification_train', train_classification, global_step=ckpt_step)
            writer.add_scalar('metrics/all_classification_val', val_classification, global_step=ckpt_step)


#### Comment out this block for IKEA ASM
            writer.add_scalar('metrics/all_kendalls_tau_train', train_kendalls_tau, global_step=ckpt_step)
            writer.add_scalar('metrics/all_kendalls_tau_val', val_kendalls_tau, global_step=ckpt_step)

            writer.add_scalar('metrics/all_phase_progression_train', train_phase_prog, global_step=ckpt_step)
            writer.add_scalar('metrics/all_phase_progression_val', val_phase_prog, global_step=ckpt_step)
########################################

        csv_dict["step"]=ckpt_step
        csv_dict["AP@5"]=ap5
        csv_dict["AP@10"]=ap10
        csv_dict["AP@15"]=ap15
        print('IMPORTANT!!!         metrics/AP@5_val', ap5, f"global_step={ckpt_step}")
        print('IMPORTANT!!!         metrics/AP@10_val', ap10, f"global_step={ckpt_step}")
        print('IMPORTANT!!!         metrics/AP@15_val', ap15, f"global_step={ckpt_step}")

#### Comment out this block for IKEA ASM
        csv_dict["Kendall Tau"]=val_kendalls_tau
        print('IMPORTANT!!!         metrics/all_kendalls_tau_val', val_kendalls_tau, f"global_step={ckpt_step}")

        csv_dict["Progression"]=val_phase_prog
        print('IMPORTANT!!!         metrics/all_phase_progression_val', val_phase_prog, f"global_step={ckpt_step}")
########################################

        # Write the row in the correct order
        with open(csv_path, 'a', newline='') as csv_file:
            csvwriter = csv.writer(csv_file)
            row = [csv_dict.get(header, '') for header in csv_headers]
            csvwriter.writerow(row)

        if writer:
            writer.flush()
            writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--dest', type=str, default='./')

    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='Cuda device to be used')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_frames', type=int, default=None, help='Path to dataset')

    args = parser.parse_args()

    if os.path.isdir(args.model_path):
        ckpts = natsorted(glob.glob(os.path.join(args.model_path, '*')))
    else:
        ckpts = [args.model_path]
    

    ckpt_mul = args.device
    main(ckpts, args)