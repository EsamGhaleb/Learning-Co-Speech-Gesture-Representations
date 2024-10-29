
from __future__ import print_function

import warnings
import warnings
from sequential_parser import get_parser
import yaml
warnings.filterwarnings('ignore')

import time
import torch
from torch import optim
from torch.utils.data import random_split
from feeders.gestures_feeder import Feeder as FeederGestures
from utils.utils import load_yaml_to_dict
from data.read_process_poses import load_keypoints_dict


import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from model.skeleton_speech_models import SupConWav2vec2GCN
from model.losses import SupConLoss, NTXent, NTXentMM
from gestures_forms_sim import measure_sim_gestures

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

torch.set_float32_matmul_precision("medium")


poses, mirrored_poses = load_keypoints_dict() 

 
class SpeechSkeletonModel(L.LightningModule):
    def __init__(self, arg):
        super().__init__()
            # build data loader
        self.arg = arg
        self.modalities = arg.model_args['modalities']
          # TODO: 
        # initialize models and losses based on opt and/or configs: unimodal, multimodal, contrastive loss
        self.model = SupConWav2vec2GCN(**arg.model_args)

        if arg.loss_function == 'ConLoss':
            self.criterion = SupConLoss(temperature=arg.temp)
        elif arg.loss_function == 'NTXent':
            self.criterion = NTXent(batch_size=arg.batch_size, n_views=2, temperature=arg.temp)
        elif arg.loss_function == "NTXentMM":
            assert len(self.modalities) == 2, "Multimodal Contrastive Loss requires 2 modalities."
        elif arg.loss_function == "Combined":
            self.uni_criterion = NTXent(batch_size=arg.batch_size, n_views=2, temperature=arg.temp)
            self.mm_criterion = NTXentMM(batch_size=arg.batch_size, temperature=arg.temp)
            self.criterion = NTXentMM(batch_size=arg.batch_size, temperature=arg.temp)
        self.optimizer = optim.SGD(self.model.parameters(),
                          lr=arg.learning_rate,
                          momentum=arg.momentum,
                          weight_decay=arg.weight_decay)
        #TODO sync batch norm
        #model = apex.parallel.convert_syncbn_model(model)

    def process_batch(self, batch):
        if "skeleton" in self.modalities:
            skeletons_1 = batch["skeleton"]["view1"]
            orig_skeletons = batch["skeleton"]["orig"] if "orig" in batch["skeleton"] else None
            skeletons_1 = batch["skeleton"]["view1"] if "view1" in batch["skeleton"] else None
            skeletons_2 = batch["skeleton"]["view2"] if "view2" in batch["skeleton"] else None
        else:
            skeletons_1 = None
            skeletons_2 = None
            orig_skeletons = None
        if "speech" in self.modalities:
            speech_1 = batch["speech"]["view1"] if "view1" in batch["speech"] else None
            orig_speech = batch["speech"]["orig"] if "orig" in batch["speech"] else None
            speech_1 = batch["speech"]["view1"] if "view1" in batch["speech"] else None
            speech_2 = batch["speech"]["view2"] if "view2" in batch["speech"] else None
            speech_lengths = batch["speech"]["lengths"] if "lengths" in batch["speech"] else None
        else:
            orig_speech = None
            speech_1 = None
            speech_2 = None
            speech_lengths = None
        if "text" in self.modalities:
            text_embeddings = batch["embeddings"]
            text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=1)
        else:
            text_embeddings = None

        label = batch["label"]
        return {
            "orig_skeletons": orig_skeletons,
            "skeletons_1": skeletons_1,
            "skeletons_2": skeletons_2,
            "orig_speech": orig_speech,
            "speech_1": speech_1,
            "speech_2": speech_2,
            "labels": label,
            "speech_lengths": speech_lengths,
            "text_embeddings": text_embeddings
        }
    
    def compute_loss_unimodal(self, features, labels=None, loss_func=None):
        if loss_func is None:
            if self.arg.loss_function == 'ConLoss':
                loss, pos, neg = self.criterion(features) if labels is None else self.criterion(features, labels)
            elif self.arg.loss_function == 'NTXent':
                loss, pos, neg = self.criterion(features)
        else:
            loss, pos, neg = loss_func(features)
        return loss, pos, neg

    def compute_loss_multimodal(self, features_1, features_2, labels=None, loss_func=None):
        if loss_func is None:
            if self.arg.loss_function == 'NTXentMM':
                loss, pos, neg = self.criterion(features_1, features_2)
        else:
            loss, pos, neg = loss_func(features_1, features_2)
        return loss, pos, neg
    
    def _shared_step(self, batch, prefix="train"):
        processed_batch = self.process_batch(batch)

        if "skeleton" in self.modalities:
            processed_batch["orig_skeletons"] = processed_batch["orig_skeletons"].float() if processed_batch["orig_skeletons"] is not None else None
            processed_batch["skeletons_1"] = processed_batch["skeletons_1"].float() if processed_batch["skeletons_1"] is not None else None
            processed_batch["skeletons_2"] = processed_batch["skeletons_2"].float() if processed_batch["skeletons_2"] is not None else None

            if processed_batch["skeletons_1"] is None and processed_batch["skeletons_2"] is None:
                skeletons = processed_batch["orig_skeletons"]
            elif processed_batch["skeletons_1"] is not None and processed_batch["skeletons_2"] is None:
                skeletons = torch.cat([processed_batch["orig_skeletons"], processed_batch["skeletons_1"]], dim=0)
            else:
                skeletons = torch.cat([processed_batch["skeletons_1"], processed_batch["skeletons_2"]], dim=0)

        if "speech" in self.modalities:
            speech_lengths = processed_batch["speech_lengths"]
            processed_batch["orig_speech"] = processed_batch["orig_speech"].float() if processed_batch["orig_speech"] is not None else None
            processed_batch["speech_1"] = processed_batch["speech_1"].float() if processed_batch["speech_1"] is not None else None
            processed_batch["speech_2"] = processed_batch["speech_2"].float() if processed_batch["speech_2"] is not None else None

            if processed_batch["speech_1"] is None and processed_batch["speech_2"] is None:
                speech = processed_batch["orig_speech"]
            elif processed_batch["speech_1"] is not None and processed_batch["speech_2"] is None:
                speech = torch.cat([processed_batch["orig_speech"], processed_batch["speech_1"]], dim=0)
            else:
                speech = torch.cat([processed_batch["speech_1"], processed_batch["speech_2"]], dim=0)
        if "text" in self.modalities:
            text_embeddings = processed_batch["text_embeddings"]

        if "speech" in self.modalities and "skeleton" in self.modalities:
            skeleton_features, speech_features = self.model(skeleton=skeletons, speech_waveform=speech, speech_lengths=speech_lengths) 
        elif "speech" in self.modalities:
            speech_features = self.model(speech_waveform=speech, speech_lengths=speech_lengths)
        elif "skeleton" in self.modalities:
            skeleton_features = self.model(skeleton=skeletons)
        
        # TODO: make labels == None if it is SSL
        if self.arg.loss_function == "Combined":
            if 'speech' in self.modalities: 
                assert (
                    processed_batch["speech_1"] is None and processed_batch["speech_2"] is None and processed_batch["orig_speech"] is not None,
                    "Unexpected speech items for combined loss"    
                )
                assert (
                    processed_batch["skeletons_1"] is not None and processed_batch["skeletons_2"] is None and processed_batch["orig_skeletons"] is not None,
                    "Unexpected skeleton items for combined loss"    
                )
            else: 
                assert 'text' in self.modalities, "Text modality is required for Combined loss if speech is not present"
                # we then just use the text embeddings as replacement of speech.
                speech_features = text_embeddings.float()
            skeleton_loss, skeleton_pos, skeleton_neg = self.compute_loss_unimodal(
                skeleton_features,
                processed_batch["labels"],
                loss_func=self.uni_criterion
            )
            #TODO randomly pick one of the views
            mm_loss, mm_pos, mm_neg = self.compute_loss_multimodal(
                speech_features,
                skeleton_features[:int(skeleton_features.shape[0] // 2), :],
                loss_func=self.mm_criterion
            )
            loss = (skeleton_loss + mm_loss) / 2
            self.log(f'{prefix}/mm_loss', mm_loss)
            self.log(f'{prefix}/mm_pos', mm_pos)
            self.log(f'{prefix}/mm_neg', mm_neg) 
            self.log(f'{prefix}/skeleton_loss', skeleton_loss)
            self.log(f'{prefix}/skeleton_pos', skeleton_pos)
            self.log(f'{prefix}/skeleton_neg', skeleton_neg)
            self.log(f'{prefix}/combined_neg', (mm_neg + skeleton_neg) / 2)
            self.log(f'{prefix}/combined_pos', (mm_pos + skeleton_pos) / 2)
            self.log(f'{prefix}/combined_loss', loss)
            self.log(f'{prefix}/loss', loss)

        elif "MM" not in self.arg.loss_function:
            skeleton_loss, skeleton_pos, skeleton_neg = self.compute_loss_unimodal(skeleton_features, processed_batch["labels"]) if "skeleton" in self.modalities else (0, 0, 0)
            speech_loss, speech_pos, speech_neg = self.compute_loss_unimodal(speech_features, processed_batch["labels"]) if "speech" in self.modalities else (0, 0, 0)
            loss = (skeleton_loss + speech_loss)/2
            self.log(f'{prefix}/loss', loss)
            self.log(f'speech/{prefix}/loss', speech_loss)
            self.log(f'skeleton/{prefix}/loss', skeleton_loss)
            self.log(f'speech/{prefix}/pos', speech_pos)
            self.log(f'skeleton/{prefix}/pos', skeleton_pos)
            self.log(f'speech/{prefix}/neg', speech_neg)
            self.log(f'skeleton/{prefix}/neg', skeleton_neg)
            pos = (skeleton_pos + speech_pos)/2
            neg = (skeleton_neg + speech_neg)/2
            self.log(f'{prefix}/pos', pos)
            self.log(f'{prefix}/neg', neg)

        else:
            loss, pos, neg = self.compute_loss_multimodal(speech_features, skeleton_features)    
            self.log(f'{prefix}/loss', loss)
            self.log(f'{prefix}/pos', pos)
            self.log(f'{prefix}/neg', neg)    
    
        return loss
    
    def training_step(self, batch, batch_idx):
        # Prepare data
        return self._shared_step(batch, "train")
        
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
        
    def test_step(self, batch, batch_idx):
        skeleton_1, skeleton_2, speech_1, speech_2, labels = self.process_batch(batch)
        
    def on_train_epoch_end(self, outputs=None) -> None:
        # print loss, time, and other info
        print('end of epoch')
        print('Current epoch: {}'.format(self.current_epoch))
        print('Current lr: {}'.format(self.optimizer.param_groups[0]['lr']))
        print('current loss is {}'.format(self.trainer.callback_metrics['train/loss']))
        
    def on_validation_epoch_end(self, outputs=None) -> None:
        correlation,difference, gestures_form = measure_sim_gestures(sup_model=self.model, processed_keypoints_dict=poses, mirrored_keypoints_dict=mirrored_poses)
        # add the correlation to the logs
        self.log('val/correlation', correlation)
        self.log('val/difference', difference)
        # pass
        
    def on_test_epoch_end(self, outputs=None) -> None:
      pass
        
    def configure_optimizers(self):
        constant_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lambda x: 1.
        )
        
        return (
            [self.optimizer],
            [
                {
                    "scheduler": constant_lr_scheduler,
                    "interval": "epoch",
                    "monitor": "val_fused_loss"
                }
            ]
        )

def main(phase='training'):
    L.seed_everything(42)

    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
   
    if p.config is not None:
        with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
                if k not in key:
                    print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    modalities = arg.model_args["modalities"]
    arg.feeder_args["modalities"] = modalities

    skeleton_augmentations = load_yaml_to_dict(arg.skeleton_augmentations_path)
   
    # Splitting the dataset
    arg.feeder_args['skeleton_augmentations'] = skeleton_augmentations
    dataset = FeederGestures(**arg.feeder_args)
    
    # Assuming you have a dataset object
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size
 
    # Splitting the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=arg.batch_size, shuffle=(train_sampler is None),
        num_workers=arg.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=arg.num_workers, 
        drop_last=True if arg.loss_function == "NTXentMM" else False
    )
    
    models_directory = 'workdir/'
    model = SpeechSkeletonModel(arg)
    # Model 

    # convert the list of modalities to a string
    modalities = "_".join(modalities)

    logger_name = arg.Experiment_name.format(modalities, arg.learning_rate, arg.batch_size, arg.temp)
    tb_logger = TensorBoardLogger(models_directory, name=logger_name)

    loggers = [tb_logger]

    if arg.wandb_entity != "none":
        experiment_info = vars(arg)
        project="CABB_" + modalities 

        wandb_logger = WandbLogger(
            config=experiment_info,
            entity=arg.wandb_entity,
            project=project,
            name=logger_name + str(time.time()), # temporary solution for unique experiment names in wandb
            id=logger_name + str(time.time())
        )
        loggers.append(wandb_logger)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    save_top_k = ModelCheckpoint(
        filename="{epoch}-{val/correlation:.2f}",
        monitor="val/correlation",
        save_top_k=10,
        every_n_epochs=1,  
        mode="max"         
    )
    save_top_k_diff = ModelCheckpoint(
        filename="{epoch}-{val/difference:.2f}",
        monitor="val/difference",
        save_top_k=10,
        every_n_epochs=1, 
        mode="max"        
    )
    save_every_5_epoch = ModelCheckpoint(
        filename="{epoch}",
        every_n_epochs=5,  
    )    

    if torch.cuda.is_available():
        # TODO: make mixed-precision optional from configs/opt to avoid errors (bf16)
        trainer = L.Trainer(
            # gradient_clip_val=0.25, 
            max_epochs=arg.num_epoch, 
            logger=loggers, 
            accelerator="gpu", 
            devices=-1, 
            num_nodes=1,
            callbacks=[
                lr_monitor, 
                save_top_k,
                save_top_k_diff,
                save_every_5_epoch
                ], 
            strategy="ddp_find_unused_parameters_true",
            enable_progress_bar=True,
            # precision="bf16",
            num_sanity_val_steps=2,
            default_root_dir=models_directory+arg.Experiment_name
            # show the progress bar
            )
    else:
        trainer = L.Trainer(
            gradient_clip_val=0.25, 
            max_epochs=arg.num_epoch, 
            logger=loggers, 
            callbacks=[lr_monitor]
            )
    if phase == 'training':
        trainer.fit(model, train_loader, val_loader)
    elif phase == 'testing':
        model = model.load_from_checkpoint('save/SupCon/CABB_tensorboard/SupCon_CABB_bimodal_lr_0.2_decay_0.0001_bsz_128_temp_0.07_trial_0/SupCon_CABB_bimodal_lr_0.2_decay_0.0001_bsz_128_temp_0.07_trial_0/version_5/checkpoints/epoch=1240-train/loss=4.85-train/loss=4.85.ckpt') 
        trainer.test(model, train_loader)
       

if __name__ == "__main__":  
    main()
