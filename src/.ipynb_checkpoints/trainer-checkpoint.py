
#python trainer.py mono-config/config-kalima.yaml --saved_model_path "saved_models/kalima_saved_models_2" --wand_project "htr-bp-kalima" --decoded_samples_file "mono_decoded_samples/decoded_samples_kalima.txt"

#python trainer.py mono-config/config-rasm.yaml --saved_model_path "saved_models/rasm_saved_models" --wand_project "htr-bp-rasm" --decoded_samples_file "mono_decoded_samples/decoded_samples_rasm.txt"

#python trainer.py mono-config/config-ohul.yaml --saved_model_path "saved_models/ohul_saved_models" --wand_project "htr-bp-ohul" --decoded_samples_file "mono_decoded_samples/decoded_samples_ohul.txt"

#python trainer.py mono-config/config-phti.yaml --saved_model_path "saved_models/phti_saved_models_2" --wand_project "htr-bp-phti" --decoded_samples_file "mono_decoded_samples/decoded_samples_phti.txt"

#python trainer.py mono-config/config-muharaf.yaml --saved_model_path "saved_models/muharaf_saved_models_2" --wand_project "htr-bp-muharaf" --decoded_samples_file "mono_decoded_samples/decoded_samples_muharaf.txt"

#python trainer.py mono-config/config-khatt.yaml --saved_model_path "saved_models/khatt_saved_models_2_2.5lr" --wand_project "htr-bp-khatt" --decoded_samples_file "mono_decoded_samples/decoded_samples_khatt.txt"

#python trainer.py mono-config/config-nust.yaml --saved_model_path "saved_models/nust_saved_models_2nd_cleand" --wand_project "htr-bp-unst" --decoded_samples_file "mono_decoded_samples/decoded_samples_nust.txt"

#python trainer.py mono-config/config-phtd.yaml --saved_model_path "saved_models/phtd_saved_models_2" --wand_project "htr-bp-phtd" --decoded_samples_file "mono_decoded_samples/decoded_samples_phtd.txt"

#python trainer.py mono-config/config-ajami.yaml --saved_model_path "saved_models/ajami_saved_models_4_concat" --wand_project "htr-bp-ajami" --decoded_samples_file "mono_decoded_samples/decoded_samples_ajami.txt"

#python trainer.py mono-config/config-unhd-all.yaml --saved_model_path "saved_models/unhd_all_saved_models" --wand_project "htr-bp-unhd-all" --decoded_samples_file "mono_decoded_samples/decoded_samples_unhd-all.txt"

#python trainer.py mono-config/config-unhd-unique.yaml --saved_model_path "saved_models/unhd_unique_saved_models" --wand_project "htr-bp-unhd-unique" --decoded_samples_file "mono_decoded_samples/decoded_samples_unhd_unique.txt"

#python trainer.py mono-config/config-khatt.yaml --saved_model_path "saved_models/khatt_saved_models_trusted_cropped_final_3nd" --wand_project "htr-bp-khatt-trusted" --decoded_samples_file "mono_decoded_samples/decoded_samples_khatt.txt"

#python trainer.py mono-config/config-phtd.yaml --saved_model_path "saved_models/phtd_saved_models_2" --wand_project "htr-bp-phtd" --decoded_samples_file "mono_decoded_samples/decoded_samples_phtd.txt"

#python trainer.py mono-config/config-phtd-all.yaml --saved_model_path "saved_models/phtd_all_saved_models" --wand_project "htr-bp-phtd" --decoded_samples_file "mono_decoded_samples/decoded_samples_phtd.txt"

#python trainer.py mono-config/config-phtd-unique.yaml --saved_model_path "saved_models/phtd_unique_saved_models" --wand_project "htr-bp-phtd" --decoded_samples_file "mono_decoded_samples/decoded_samples_phtd.txt"

#python trainer.py mono-config/config-phtd-final.yaml --saved_model_path "saved_models/phtd_final_saved_models" --wand_project "htr-bp-phtd" --decoded_samples_file "mono_decoded_samples/decoded_samples_phtd.txt"



#python trainer.py mono-config/config-ajami.yaml --saved_model_path "saved_models/ajami_saved_models_no_clean" --wand_project "htr-bp-ajami" --decoded_samples_file "mono_decoded_samples/decoded_samples_ajami.txt"  --metrics_csv "training_metrics_ajami_no_clean.csv"

#python trainer.py mono-config/config-muharaf.yaml --saved_model_path "saved_models/muharaf_saved_models_2_cleaned_resize" --wand_project "htr-bp-muharaf" --decoded_samples_file "mono_decoded_samples/decoded_samples_muharaf_cleaned.txt"
#python trainer.py mono-config/config-muharaf.yaml --saved_model_path "saved_models/muharaf_saved_models_2_no_resize" --wand_project "htr-bp-muharaf" --decoded_samples_file "mono_decoded_samples/decoded_samples_muharaf_cleaned.txt"

#python trainer.py mono-config/config-muharaf.yaml --saved_model_path "saved_models/muharaf_saved_models_2_cleaned_resize" --wand_project "htr-bp-muharaf" --decoded_samples_file "mono_decoded_samples/decoded_samples_muharaf_cleaned.txt"
#python trainer.py mono-config/config-muharaf.yaml --saved_model_path "saved_models/muharaf_saved_models_2_cleaned_resize_keep_eng" --wand_project "htr-bp-muharaf" --decoded_samples_file "mono_decoded_samples/decoded_samples_muharaf_cleaned.txt"

#python trainer.py mono-config/config-ajami.yaml --saved_model_path "saved_models/ajami_saved_models_no_share_clean_by_exclude" --wand_project "htr-bp-ajami" --decoded_samples_file "mono_decoded_samples/decoded_samples_ajami.txt"  

#python trainer.py mono-config/config-ajami.yaml --saved_model_path "saved_models/ajami_saved_models_no_share" --wand_project "htr-bp-ajami" --decoded_samples_file "mono_decoded_samples/decoded_samples_ajami.txt"  

#python trainer.py mono-config/config-muharaf.yaml --saved_model_path "saved_models/muharaf_saved_models_2_cleaned_resize-2nd-round" --wand_project "htr-bp-muharaf" --decoded_samples_file "mono_decoded_samples/decoded_samples_muharaf_cleaned.txt"

#python trainer.py mono-config/config-phti.yaml --saved_model_path "saved_models/phti_saved_models_2_cleaned" --wand_project "htr-bp-phti" --decoded_samples_file "mono_decoded_samples/decoded_samples_phti.txt"
import wandb
import argparse
from omegaconf import OmegaConf

import sys
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.htr_dataset_modified import HTRDataset
import csv
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()


from torch.utils.data._utils.collate import default_collate

def custom_collate_fn(batch):
    imgs, transcrs = zip(*batch)
    imgs = default_collate(imgs)  # stack images into one tensor
    return imgs, transcrs         # keep transcriptions as list of lists


from models import HTRNet
from utils.transforms import aug_transforms

import torch.nn.functional as F

from utils.metrics import CER, WER

class HTRTrainer(nn.Module):
    def __init__(self, config, saved_model_path, wand_project, decoded_samples_file, metrics_csv_path):
        super(HTRTrainer, self).__init__()
        self.config = config
        
        self.saved_model_path = saved_model_path
        self.wand_project = wand_project
        self.decoded_samples_file = decoded_samples_file
        self.metrics_csv_path = Path(metrics_csv_path)
        self._init_metrics_csv()
        
        self.prepare_dataloaders()
        self.prepare_net()
        self.prepare_losses()
        self.prepare_optimizers()

    def _init_metrics_csv(self):
        """Create CSV with header if it doesn't exist."""
        if not self.metrics_csv_path.exists():
            self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "split", "loss", "cer", "wer", "lr"])

    def _append_metrics_row(self, epoch, split, loss, cer, wer):
        """Append one row for val/test to CSV."""
        try:
            lr = float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            lr = None
        with self.metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                int(epoch),
                str(split),
                None if loss is None else float(loss),
                None if cer  is None else float(cer),
                None if wer  is None else float(wer),
                None if lr   is None else float(lr)
            ])
            
    def prepare_dataloaders(self):
    

        config = self.config

        # prepare datset loader
        dataset_folder = config.data.path
        fixed_size = (config.preproc.image_height, config.preproc.image_width)

        #train_set = HTRDataset(dataset_folder, 'train', fixed_size=fixed_size, transforms=aug_transforms)
        
        #classes = train_set.character_classes

        train_set = HTRDataset(dataset_folder, 'train', fixed_size=fixed_size, transforms=aug_transforms)
        val_set = HTRDataset(dataset_folder, 'val', fixed_size=fixed_size, transforms=None)
        test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size, transforms=None)

        print('# training lines ' + str(len(train_set)))
        print('# validation lines ' + str(len(val_set)))
        print('# testing lines ' + str(len(test_set)))

        # Build full character set from all splits
        all_classes = set(train_set.character_classes)
        all_classes.update(val_set.character_classes)
        all_classes.update(test_set.character_classes)

        all_classes.add(' ')   # ✅ add directly to the set

        # Convert to sorted unique list
        classes = np.array(sorted(all_classes))  # or np.unique(list(all_classes))
        
        
        #print('# training lines ' + str(train_set.__len__()))

        #val_set = HTRDataset(dataset_folder, 'val', fixed_size=fixed_size, transforms=None)
        #print('# validation lines ' + str(val_set.__len__()))

        #test_set = HTRDataset(dataset_folder, 'test', fixed_size=fixed_size, transforms=None)
        #print('# testing lines ' + str(test_set.__len__()))

        # augmentation using data sampler
        train_loader = DataLoader(train_set, batch_size=config.train.batch_size, 
                                  shuffle=True, num_workers=config.train.num_workers)
        if val_set is not None:
            val_loader = DataLoader(val_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)
        test_loader = DataLoader(test_set, batch_size=config.eval.batch_size,  
                                    shuffle=False, num_workers=config.eval.num_workers)

        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        # add space to classes, if not already there
        classes = np.unique(classes)

        # save classes in data folder
        #np.save(os.path.join(dataset_folder, 'classes_cleaned.npy'), classes)
        np.save(os.path.join(dataset_folder, 'classes.npy'), classes)
        #np.save(os.path.join(dataset_folder, 'classes_wrong_labels.npy'), classes)
        

        # create dictionareies for character to index and index to character 
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

    def prepare_losses(self):
        self.ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) /self.config.train.batch_size

    def prepare_optimizers(self):
        config = self.config
        optimizer = torch.optim.AdamW(self.net.parameters(), config.train.lr, weight_decay=0.00005)

        self.optimizer = optimizer

        max_epochs = config.train.num_epochs
        if config.train.scheduler == 'mstep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])
        else:
            raise NotImplementedError('Alternative schedulers not implemented yet')

    def decode(self, tdec, tdict, blank_id=0):
  # Ensure tdec is iterable
        if isinstance(tdec, (int, np.integer)):
            tdec = [int(tdec)]
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([tdict[t] for t in tt if t != blank_id])
        
        #reverse predictions
        dec_transcr = dec_transcr[::-1]
        
        return dec_transcr
                
    def sample_decoding(self):

        # get a random image from the test set
        img, transcr = self.loaders['val'].dataset[np.random.randint(0, len(self.loaders['val'].dataset))]

        img = img.unsqueeze(0).to(self.config.device)

        self.net.eval()
        with torch.no_grad():
            tst_o = self.net(img)
            if self.config.arch.head_type == 'both':
                tst_o = tst_o[0]

        self.net.train()

        tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        # remove duplicates
        dec_transcr = self.decode(tdec, self.classes['i2c'])


         
        orig_display = transcr[::-1].strip()
        pred_display = dec_transcr.strip()
        
        
        print('orig:: ' + orig_display)
        print('pred:: ' + pred_display)
        
        #  append to .txt file
        #file_path = "decoded_samples_muharaf.txt"
        file_path = self.decoded_samples_file
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {self.current_epoch}\n")
            f.write(f"Original: {orig_display}\n")
            f.write(f"Predicted: {pred_display}\n\n")
            

    def train(self, epoch):
        self.current_epoch = epoch
        config = self.config
        device = config.device

        self.net.train()
        
        total_loss = 0
        num_batches = 0
        
        t = tqdm.tqdm(self.loaders['train'])
        t.set_description('Epoch {}'.format(epoch))
        for iter_idx, (img, transcr) in enumerate(t):
            self.optimizer.zero_grad()

            img = img.to(device)

            if config.arch.head_type == "both":
                output, aux_output = self.net(img)
            else:
                output = self.net(img)

            act_lens = torch.IntTensor(img.size(0)*[output.size(0)])
            labels = torch.IntTensor([self.classes['c2i'][c] for c in ''.join(transcr)])
            label_lens = torch.IntTensor([len(t) for t in transcr])

            act_lens = act_lens.to(output.device)
            label_lens = label_lens.to(output.device)
            labels = labels.to(output.device)

            assert output.shape[1] == img.size(0), "Batch mismatch between model output and input"
            assert sum(label_lens) == len(labels), "Sum of label lengths doesn't match label tensor size"
            assert (act_lens >= label_lens).all(), "CTC constraint violated: input length < target length"

            
            loss_val = self.ctc_loss(output, labels, act_lens, label_lens)

            if config.arch.head_type == "both":
                loss_val += 0.1 * self.ctc_loss(aux_output, labels, act_lens, label_lens)

            tloss_val = loss_val.item()

            total_loss += tloss_val
            num_batches += 1
        
            loss_val.backward()
            self.optimizer.step()    

            t.set_postfix(values='loss : {:.2f}'.format(tloss_val))
            
            #wandb.log({"train/loss_batch": tloss_val, "epoch": epoch})

        avg_loss = total_loss / num_batches
        wandb.log({"epoch": epoch,"train/loss": avg_loss})
        
        #self.sample_decoding()
    
    def test(self, epoch, tset='val'):

        config = self.config
        device = config.device

        self.net.eval()

        if tset=='test':
            loader = self.loaders['test']
        elif tset=='val':
            loader = self.loaders['val']
        else:
            print("not recognized set in test function")

        print('####################### Evaluating {} set at epoch {} #######################'.format(tset, epoch))
        
        cer, wer = CER(), WER(mode=config.eval.wer_mode)
        
        total_loss = 0
        num_batches = 0
        
        for (imgs, transcrs) in tqdm.tqdm(loader):

            imgs = imgs.to(device)
            with torch.no_grad():
                o = self.net(imgs)
            # if o tuple keep only the first element
            if config.arch.head_type == 'both':
                o = o[0]
            
            tdecs = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
            ####to calculate validaton loss 
            act_lens = torch.IntTensor(imgs.size(0) * [o.size(0)]).to(device)
            label_lens = torch.IntTensor([len(t) for t in transcrs]).to(device)
            labels = torch.IntTensor([self.classes['c2i'][c] for c in ''.join(transcrs)]).to(device)

            val_loss = self.ctc_loss(o, labels, act_lens, label_lens)
            total_loss += val_loss.item()
            num_batches += 1

            
            for tdec, transcr in zip(tdecs, transcrs):
                transcr = transcr.strip()
                dec_transcr = self.decode(tdec, self.classes['i2c']).strip()
                cer.update(dec_transcr, transcr[::-1])
                wer.update(dec_transcr, transcr[::-1])
                #cer.update(dec_transcr, transcr)
                #wer.update(dec_transcr, transcr)
        
        cer_score = cer.score()
        wer_score = wer.score()

        print('CER at epoch {}: {:.3f}'.format(epoch, cer_score))
        print('WER at epoch {}: {:.3f}'.format(epoch, wer_score))
        avg_loss = total_loss / num_batches

        #wandb.log({f"{tset}/CER": cer.score(),f"{tset}/WER": wer.score(),"epoch": epoch})
        #wandb.log({f"{tset}/loss": avg_loss,f"{tset}/CER": cer.score(),f"{tset}/WER": wer.score(),"epoch": epoch})
        wandb.log({"epoch": epoch,f"{tset}/loss": avg_loss,f"{tset}/CER": cer.score(),f"{tset}/WER":wer.score()})
        self._append_metrics_row(epoch=epoch, split=tset, loss=avg_loss, cer=cer_score, wer=wer_score)

        self.net.train()

    def save(self, epoch):
        print('####################### Saving model at epoch {} #######################'.format(epoch))
        
        if not os.path.exists(self.saved_model_path):
            os.makedirs(self.saved_model_path)
        torch.save(self.net.cpu().state_dict(), os.path.join(self.saved_model_path, f"htrnet_{epoch}.pt"))

        self.net.to(self.config.device)


def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

parser = argparse.ArgumentParser()
parser.add_argument("config_file", help="Path to config YAML file")  # positional arg

parser.add_argument("--saved_model_path", default="./saved_models", help="Path to save trained models", type=str)
parser.add_argument("--wand_project", default="htr-bp", help="W&B project name", type=str)
parser.add_argument("--decoded_samples_file", default="decoded_samples.txt", help="Output file for decoded samples", type=str)
parser.add_argument("--metrics_csv", default="training_metrics_ajami_no_clean.csv",
                    help="CSV to log per-epoch metrics for val/test (epoch, split, loss, CER, WER, LR)",
                    type=str)
args = parser.parse_args()


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    #config = parse_args()
    # -------- Load Hydra/OmegaConf config --------
    config = OmegaConf.load(args.config_file)
    OmegaConf.set_struct(config, True)

    max_epochs = config.train.num_epochs

# ----------------------- adding wandb integration--------------------    
    wandb.init(
        project=args.wand_project,  # ✅  new project name
        entity="sana-ltu", # ✅  username or team
        config=OmegaConf.to_container(config, resolve=True),
        name="htr-run-cropped-final-corrected-labels-3rd")
#-------------------- -------------------- -------------------- ---------    

   
    htr_trainer = HTRTrainer(config, args.saved_model_path, args.wand_project, args.decoded_samples_file,args.metrics_csv)
    cnt = 1
    print('Training Started!')
    htr_trainer.test(0, 'val')
    for epoch in range(1, max_epochs + 1):

        htr_trainer.train(epoch)
        htr_trainer.scheduler.step()

        # save and evaluate the current model
        htr_trainer.test(epoch, 'val')
        if epoch % config.train.save_every_k_epochs == 0:
            htr_trainer.save(epoch)
            #htr_trainer.test(epoch, 'val')
            htr_trainer.test(epoch, 'test')

    # save the final model
    if not os.path.exists(args.saved_model_path):
        os.makedirs(args.saved_model_path)
    torch.save(htr_trainer.net.cpu().state_dict(), os.path.join(args.saved_model_path, config.save))
    