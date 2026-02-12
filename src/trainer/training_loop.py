import copy
import functools
import os
import time
import re
import numpy as np
import torch 
from torch.optim import AdamW
from os.path import join as pjoin
from tqdm import tqdm 

# Third-party/Local imports
import blobfile as bf
from diffusion import logger
from utils import dist_utils
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler, create_named_schedule_sampler
from utils.sampler_utils import ClassifierFreeSampleModel
from data.get_data import get_dataset_loader 

# Assuming these exist based on your original code
# from some_module import evaluation, generate, sample_goal, get_target_location, load_model_wo_clip

class TrainLoop:
    def __init__(self, cfg, train_platform, model, diffusion, data):
        self.cfg = cfg
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.data = data
        
        # --- Config Mapping (Matching YAML Structure) ---
        self.dataset_name = self.cfg.data.dataset
        self.save_dir = self.cfg.experiment.save_dir
        self.overwrite = self.cfg.experiment.overwrite
        
        # Training Params
        self.batch_size = self.cfg.data.batch_size
        self.lr = self.cfg.training.lr
        self.log_interval = self.cfg.training.log_interval
        self.save_interval = self.cfg.training.save_interval
        self.resume_checkpoint = self.cfg.training.resume_checkpoint
        self.weight_decay = self.cfg.training.weight_decay 
        self.lr_anneal_steps = self.cfg.training.lr_anneal_steps
        self.num_steps = self.cfg.training.num_steps
        self.use_ema = self.cfg.training.use_ema
        
        # Model/Data Params
        self.cond_mode = model.cond_mode # Often derived from model class, but can be in cfg.model.cond_mode
        self.parts_to_model = self.cfg.data.parts
        
        # EMA Setup
        self.model_avg = None
        if self.use_ema:
            self.model_avg = copy.deepcopy(self.model)
            
        # Evaluation Model Selection
        # NOTE: Fixed bug where 'self.args' was used instead of 'self.cfg' or 'self'
        self.model_for_eval = self.model_avg if self.use_ema else self.model
        
        # Guidance Setup
        if self.cfg.sampling.guidance_param != 1:
            self.model_for_eval = ClassifierFreeSampleModel(self.model_for_eval)

        self.step = 0
        self.resume_step = 0 
        self.global_batch = self.batch_size 
        self.num_epochs = self.num_steps // len(self.data) + 1
        
        self.json_path_list = [] # Seems unused or populated later
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        
        self.mp_trainer = MixedPrecisionTrainer(model=self.model, use_fp16=False, fp16_scale_growth=1e-3)

        if self.use_ema:
            self.opt = AdamW(
                self.mp_trainer.master_params, 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                betas=(0.9, self.cfg.training.adam_beta2)
            )
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )

        if self.resume_step:
            self._load_optimizer_state()
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_utils.dev() != 'cpu':
            self.device = torch.device(dist_utils.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        # Dataset Loader
        if self.dataset_name in ['realsense-mmb-walk', 'toy']:
            # self.data is passed in init, but here we seem to reload or check logic?
            # Keeping original logic structure but ensuring variables match
            pass 
            # trainloader = get_dataset_loader(self.json_path_list, self.parts_to_model) 
        else:
            raise ValueError(f"Dataset {self.dataset_name} not found")
    

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # Adjust step to resume correctly
            self.step = self.resume_step 

            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            
            state_dict = dist_utils.load_state_dict(
                resume_checkpoint, map_location=dist_utils.dev())
            
            if 'model_avg' in state_dict:
                print("loading both model and model avg")
                state_dict_model, state_dict_avg = state_dict['model'], state_dict['model_avg']
                
                # Assuming load_model_wo_clip is imported
                load_model_wo_clip(self.model, state_dict_model)
                load_model_wo_clip(self.model_avg, state_dict_avg)
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.use_ema:
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict(), strict=False)

    def _load_optimizer_state(self):
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt")

        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_utils.load_state_dict(
                opt_checkpoint, map_location=dist_utils.dev()
            )
            
            # Preserve target weight decay from config
            tgt_wd = self.opt.param_groups[0]['weight_decay']
            self.opt.load_state_dict(state_dict)
            
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd
            self.opt.param_groups[0]['capturable'] = True 

    def cond_modifiers(self, cond, motion):
        self.target_cond_modifier(cond, motion)
    
    def target_cond_modifier(self, cond, motion):
        if self.cfg.model.multi_target_cond:
            batch_size = motion.shape[0]
            # Assuming sample_goal and get_target_location are imported
            cond['target_joint_names'], cond['is_heading'] = sample_goal(
                batch_size, motion.device, self.cfg.data.target_joint_names
            )

            cond['target_cond'] = get_target_location(
                motion, 
                self.data.dataset.mean[None, :, None, None], 
                self.data.dataset.std[None, :, None, None],
                cond['lengths'], 
                self.data.dataset.t2m_dataset.opt.joints_num, 
                self.model.all_goal_joint_names, 
                cond['target_joint_names'], 
                cond['is_heading']
            ).detach()

    def run_loop(self):
        print('train steps:', self.num_steps)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break

                self.cond_modifiers(cond['y'], motion)
                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond)

                if self.total_step() % self.log_interval == 0:
                    for k, v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))
                        
                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')
                
                if self.total_step() % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    if self.use_ema:
                        self.model_avg.eval()
                    
                    self.evaluate()
                    self.generate_during_training()
                    
                    self.model.train()
                    if self.use_ema:
                        self.model_avg.train()

                    # Safety break for tests
                    if os.environ.get('DIFFUSION_TRAINING_TEST', "") and self.total_step() > 0:
                        return

                self.step += 1
            
            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break
    
    def evaluate(self):
        if not self.cfg.evaluation.eval_during_training:
            return
        
        start_eval = time.time()

        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_{self.dataset_name}_{(self.total_step()):09d}.log')
            diversity_times = 300
            mm_num_times = 0
            
            # Assuming evaluation function is imported
            eval_dict = evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file, 
                replication_times=self.cfg.evaluation.eval_rep_times, 
                diversity_times=diversity_times, 
                mm_num_times=mm_num_times, 
                run_mm=False
            )

            print(eval_dict)

            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i], iteration=self.total_step(), group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Eval')
            
            end_eval = time.time()
            print(f"Evaluation time: {round(end_eval - start_eval)/60} min")

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self.update_average_model()
        self._anneal_lr()
        self.log_step()

    def update_average_model(self):
        if self.use_ema:
            params = self.mp_trainer.master_params
            for param, avg_params in zip(params, self.model_avg.parameters()):
                avg_params.data.mul_(self.cfg.training.avg_model_beta).add_(param.data, alpha=1 - self.cfg.training.avg_model_beta)

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.batch_size):
            # NOTE: Variable 'last_batch' was calculated but not used in logic except implicit checks
            # Adjusted for potential DDP usage if added later
            
            t, weights = self.schedule_sampler.sample(batch.shape[0], dist_utils.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses, 
                self.model, # Changed ddp_model to model as ddp_model wasn't defined in init
                batch, 
                t, 
                model_kwargs=cond, 
                dataset=self.data.dataset
            )

            losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses['loss'].detach())
            
            loss = (losses['loss'] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
    
    def log_step(self):
        logger.logkv("step", self.total_step())
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.total_step()):09d}.pt"
    
    def generate_during_training(self):
        if not self.cfg.sampling.gen_during_training:
            return
        
        # Deepcopy the whole config to avoid modifying the original
        gen_args = copy.deepcopy(self.cfg)
        
        # We manually flatten or set specific args if the 'generate' function expects a flat namespace
        # (Assuming generate function handles this object)
        gen_args.model_path = os.path.join(self.save_dir, self.ckpt_file_name())
        gen_args.output_dir = os.path.join(self.save_dir, f'{self.ckpt_file_name()}.samples')
        
        # Mapping from nested structure to what generate() likely expects if it wasn't updated
        # If generate() is updated, these assignments can be cleaned up
        gen_args.num_samples = self.cfg.sampling.gen_num_samples
        gen_args.num_repetitions = self.cfg.sampling.gen_num_repetitions
        gen_args.guidance_param = self.cfg.sampling.guidance_param
        gen_args.motion_length = 6
        gen_args.input_text = ''
        gen_args.text_prompt = '' 
        gen_args.action_file = '' 
        gen_args.action_name = '' 
        gen_args.dynamic_text_path = ''
        
        if self.cfg.model.multi_target_cond:
            gen_args.sampling_mode = 'goal'
            gen_args.target_joint_source = 'data'
            
        # Assuming generate is imported
        all_sample_save_path = generate(gen_args)
        self.train_platform.report_media(title='Motion', series='Predicted Motion', iteration=self.total_step(), local_path=all_sample_save_path)

    def find_resume_checkpoint(self):
        if not os.path.exists(self.save_dir):
            return None
        matches = {file: re.match(r'model(\d+).pt', file) for file in os.listdir(self.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}
        return pjoin(self.save_dir, models[max(models)]) if models else None

    def total_step(self):
        return self.step + self.resume_step
    
    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
                for e in clip_weights:
                    del state_dict[e]
            
            state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
            del_clip(state_dict)

            if self.use_ema:
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model ...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)
            
        save_checkpoint()

        with bf.BlobFile(bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"), "wb") as f:
            opt_state = self.opt.state_dict()
            torch.save(opt_state, f)

# Helper functions usually kept outside the class
def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)