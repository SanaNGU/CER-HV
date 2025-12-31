#for zero shot evaluation using the pre-trained multiligual trained model using same classes used in tne original pretraining by: 
#python evaluate.py multi_config/config-nust.yaml   #this will do zero shot using resme path in the config tested on nust using the 
#python evaluate.py multi_config/config-khatt.yaml
#python evaluate.py multi_config/config-phtd.yaml

#for mono evaluation
#python evaluate.py multi_config/config-khatt.yaml
#python evaluate.py multi_config/config-muharaf.yaml
#python evaluate.py multi_config/config-nust.yaml
#python evaluate.py multi_config/config-phtd.yaml
#python evaluate.py multi_config/config-phti.yaml
#python evaluate.py multi_config/config-ajami.yaml
import argparse
from omegaconf import OmegaConf
import wandb
import sys
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.htr_dataset_modified import HTRDataset
import pandas as pd

from models import HTRNet
from utils.metrics import CER, WER

class HTREval(nn.Module):
    def __init__(self, config):
        super(HTREval, self).__init__()
        self.config = config

        self.prepare_dataloaders()
        self.prepare_net()

    def prepare_dataloaders(self):

        config = self.config

        # prepare datset loader
        dataset_folder = config.data.path
        fixed_size = (config.preproc.image_height, config.preproc.image_width)
        #--------------------only to test wrong annotation----remove later--------
        train_set = HTRDataset(dataset_folder, 'train', fixed_size=fixed_size, transforms=None)
        #-----------------------------------------------
        print('# training  lines ' + str(train_set.__len__()))
        val_set = HTRDataset(dataset_folder, 'val', fixed_size=fixed_size, transforms=None)
        print('# validation lines ' + str(val_set.__len__()))

        test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size, transforms=None)
        print('# testing lines ' + str(test_set.__len__()))
#-----------------------------------------------------------------------------------------------
        # load classes from the training set saved in the data folder
        #classes = np.load(os.path.join(dataset_folder, 'classes.npy'))
        if hasattr(config.eval, "model_classes_path") and config.eval.model_classes_path:
            print(f"🔤 Using model classes from: {config.eval.model_classes_path}")
            classes = np.load(config.eval.model_classes_path)
        else:
            print("🔤 Using dataset's own classes.npy")
            classes = np.load(os.path.join(dataset_folder, 'classes.npy'))
            #classes = np.load(os.path.join(dataset_folder, 'classes_cleaned.npy'))

#-------------------------------------------------------------------------------------       
        #--------------------only to test wrong annotation----remove later--------

        train_loader = DataLoader(train_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)
        #-----------------------------------------------

        val_loader = DataLoader(val_set, batch_size=config.eval.batch_size,
                                shuffle=False, num_workers=config.eval.num_workers)

        test_loader = DataLoader(test_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)

       #self.loaders = {'val': val_loader, 'test': test_loader}
        
        #--------------------only to test wrong annotation----remove later--------

        self.loaders = {'train': train_loader,'val': val_loader, 'test': test_loader}
        #-------------------------------------------------------------------------------

        # create dictionaries for character to index and index to character 
        # 0 index is reserved for CTC blank
        cdict = {c:(i+1) for i,c in enumerate(classes)}
        icdict = {(i+1):c for i,c in enumerate(classes)}

        self.classes = {
            'classes': classes,
            'c2i': cdict,
            'i2c': icdict
        }

    def prepare_net(self):

        config = self.config

        device = config.device

        print('Preparing Net - Architectural elements:')
        print(config.arch)

        classes = self.classes['classes']

        net = HTRNet(config.arch, len(classes) + 1)
        
        if config.resume is not None:
            print('resuming from checkpoint: {}'.format(config.resume))
            load_dict = torch.load(config.resume)
            load_status = net.load_state_dict(load_dict, strict=True)
            print(load_status)
        net.to(device)

        # print number of parameters
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Number of parameters: {}'.format(n_params))

        self.net = net

    def decode(self, tdec, tdict, blank_id=0):
        if isinstance(tdec, (int, np.integer)):
            tdec = [int(tdec)]        
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        dec_transcr = dec_transcr[::-1]
        
        return dec_transcr
        # use it only when infer on ajami train
    #def decode(self, tdec, tdict, blank_id=0, reverse_output=True):
        #import numpy as np
       ########## # ensure iterable
        #if isinstance(tdec, (int, np.integer)):
           # tdec = [tdec]
        #elif not hasattr(tdec, "__iter__"):
            #tdec = list(tdec)
        #tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        #decoded = ''.join([tdict[t] for t in tt if t != blank_id])
        #return decoded[::-1] if reverse_output else decoded

    def load_allowed_charset(self, allowed_classes_path=None):
        """
        Loads a language-specific allowed charset and returns allowed indices.
        If not provided, it defaults to the dataset's own classes.
        """
        if allowed_classes_path is None:
            # default = same as current dataset
            allowed_chars = self.classes['classes']
        else:
            allowed_chars = np.load(allowed_classes_path)

        allowed_ids = [
            self.classes['c2i'][c]
            for c in allowed_chars if c in self.classes['c2i']
        ]

        print(f"✅ Restricting decoding to {len(allowed_ids)} allowed characters.")
        return torch.tensor(allowed_ids, dtype=torch.long)

        

    def test(self, epoch, tset='test'):

        config = self.config
        device = config.device

        self.net.eval()

        if tset=='test':
            loader = self.loaders['test']
        elif tset=='val':
            loader = self.loaders['val']
        elif tset=='train':
            loader = self.loaders['train']
        else:
            print("not recognized set in test function")

        print('####################### Evaluating {} set at epoch {} #######################'.format(tset, epoch))
        
        cer, wer = CER(), WER(mode=config.eval.wer_mode)
        rows = []  # to collect GT/predictions for CSV
        txt_lines = []
        for (imgs, transcrs) in tqdm.tqdm(loader):

            imgs = imgs.to(device)
            with torch.no_grad():
                o = self.net(imgs)
            # if o tuple keep only the first element
            if config.arch.head_type == 'both':
                o = o[0]


            # --- [NEW] Restrict decoding to target-language charset ---
            # To restrict to KHATT, set this path manually or pass via config
            allowed_classes_path = getattr(config.eval, "allowed_classes_path", None)

            if allowed_classes_path is not None:
                allowed_ids = self.load_allowed_charset(allowed_classes_path).to(o.device)
                mask = torch.full_like(o, -float('inf'))
                mask[:, :, allowed_ids] = 0
                o = o + mask  # apply mask before argmax
            # -----------------------------------------------------------

            tdecs = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()

            for tdec, transcr in zip(tdecs, transcrs):
                transcr = transcr.strip()
                dec_transcr = self.decode(tdec, self.classes['i2c']).strip()

                #cer.update(dec_transcr, transcr)
                #wer.update(dec_transcr, transcr)

                # ✅ Print both in natural Arabic
                print("GT:", transcr[::-1])
                print("Prediction:", dec_transcr)

                # ✅ Add to CSV data
                rows.append({
                 "GroundTruth": transcr[::-1],
                 "Prediction": dec_transcr
                    })
                txt_lines.append(f"gt: {transcr[::-1]}")
                txt_lines.append(f"prediction: {dec_transcr}")
                txt_lines.append("")  # blank line separator
               
# --------------------------------------
                cer.update(dec_transcr, transcr[::-1])
                wer.update(dec_transcr, transcr[::-1])
        
        cer_score = cer.score()
        wer_score = wer.score()

        print('CER at epoch {}: {:.5f}'.format(epoch, cer_score))
        print('WER at epoch {}: {:.5f}'.format(epoch, wer_score))
        # ---------- Save outputs ----------
        dataset_tag = "khatt"
        out_dir = "error_analysis"

        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"{dataset_tag}_{tset}_results_epoch_{epoch}.csv")

        #csv_path = f"{tset}_results_epoch_{epoch}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved results to {csv_path}")
                   
        csv_path = f"{dataset_tag}_{tset}_results_epoch_{epoch}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved results to {csv_path}")


        # NEW: write the plain-text predictions file
        txt_path = os.path.join(out_dir, f"{dataset_tag}_{tset}_predictions_epoch_{epoch}.txt")

        #txt_path = f"{dataset_tag}_{tset}_predictions_epoch_{epoch}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
        print(f"Saved GT/predictions log to {txt_path}")
        # ----------------------------------
#--------------------------------------------------------------------------------        
        self.net.train()


def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()
    max_epochs = config.train.num_epochs

    #wandb.init( project="htr-bp-khatt",              # same project used in training
        #entity="sana-ltu",             # your username
        #name="eval-kn",               # custom name for this run
        #config=OmegaConf.to_container(config, resolve=True)
    #)
    
    htr_eval = HTREval(config)
    #htr_eval.test(0, 'test')

    #htr_eval.test(0, 'train')
    htr_eval.test(0, 'val')
    