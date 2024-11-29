import logging
import os
import re
import numpy as np
import torch
from tqdm import tqdm

from .trainer import SparseFineTuner

logger = logging.getLogger(__name__)


def LotteryTicketSparseFineTuner(_Trainer):

    _SparseFineTuner = SparseFineTuner(_Trainer)

    class _LotteryTicketSparseFineTuner(_SparseFineTuner):

        def __init__(self, unfreeze_strategy='global', **kwargs): ## unfreeze_strategy: 'global' or 'layer_wise_selection'
            super().__init__(**kwargs)
            logger.setLevel(self.args.get_process_log_level())
            if self.sft_args.ft_params_num is None:
                self.n_tunable_params = int(
                    self.sft_args.ft_params_proportion * self._num_maskable_params
                )
            else:
                self.n_tunable_params = self.sft_args.ft_params_num
            self.unfreeze_strategy = unfreeze_strategy
            self.selected_layers = kwargs.get('selected_layers', []) ## prefix of the layer names which are selected for fine-tuning, if nothing provided then assume that all layers are allowed to be fine-tuned 
            if self.unfreeze_strategy == 'layer_wise_selection':
                self.per_layer_percentage = kwargs.get('per_layer_percentage') ## dictionary
                self.per_layer_parameters_tuned = {}
                for pattern_str, percentage in self.per_layer_percentage.items():
                    self.per_layer_parameters_tuned[pattern_str] = int(percentage * self.n_tunable_params)


        def unfreeze_k_most_changed_params(self, k):
            with torch.no_grad():
                diffs = []
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Finding masking threshold',
                    disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    p.grad = None  # save some memory
                    if n in self.maskable_params:
                        # If selected_layers is specified, check if the parameter's layer is in selected_layers
                        if not self.selected_layers or n.startswith(tuple(self.selected_layers)):
                            delta = p - self._original_params[n].to(p.device)
                            delta = delta.view(-1)
                            valid_indices = (~self._mask[n]).view(-1)
                            valid_deltas = delta[valid_indices]
                            abs_deltas = torch.abs(valid_deltas)
                            diffs.extend(abs_deltas.tolist())

                if k > len(diffs):
                    raise ValueError(
                        f'Was requested to unfreeze {k} params, but only {len(diffs)} are frozen in the selected scope.'
                    )
                diffs = np.partition(diffs, len(diffs) - k)
                thresh = diffs[len(diffs) - k]
                logger.info(f'Masking threshold = {thresh}')

                n_masked = 0
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Updating masks',
                    disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    if n in self.maskable_params:
                        if not self.selected_layers or n.startswith(tuple(self.selected_layers)):
                            abs_delta = (p - self._original_params[n].to(p.device)).abs()
                            to_mask = (abs_delta >= thresh) & (~self._mask[n])
                            self._mask[n] = to_mask | self._mask[n]
                            n_masked += to_mask.sum()
                logger.info(f'Unmasked {n_masked} params')

        # New function for Variant 2: Per-Layer Percentage
        def unfreeze_k_most_changed_params_per_layer_percentage(self):
            with torch.no_grad():
                for pattern_str, total_tunable in self.per_layer_parameters_tuned.items():
                    n_tunable = total_tunable // self.sft_args.n_ft_iterations
                    diffs = []
                    pattern = re.compile(pattern_str)
                    for n, p in tqdm(
                        list(self.model.named_parameters()),
                        desc=f'Finding masking threshold for {pattern_str}',
                        disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                    ):
                        p.grad = None
                        if n in self.maskable_params and pattern.match(n):
                            delta = p - self._original_params[n].to(p.device)
                            delta = delta.view(-1)
                            valid_indices = (~self._mask[n]).view(-1)
                            valid_deltas = delta[valid_indices]
                            abs_deltas = torch.abs(valid_deltas)
                            diffs.extend(abs_deltas.tolist())
                            logger.info(f'Getting deltas for layer {n}')      

                    if n_tunable > len(diffs):
                        raise ValueError(
                            f'Was requested to unfreeze {n_tunable} params in {pattern_str}, but only {len(diffs)} are frozen in the selected scope.'
                        )
                    diffs = np.partition(diffs, len(diffs) - n_tunable)
                    thresh = diffs[len(diffs) - n_tunable]
                    logger.info(f'Masking threshold for {pattern_str} = {thresh}')   
                    n_masked = 0
                    for n, p in tqdm(
                        list(self.model.named_parameters()),
                        desc=f'Updating masks for {pattern_str}',
                        disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                    ):
                        if n in self.maskable_params and pattern.match(n):
                            abs_delta = (p - self._original_params[n].to(p.device)).abs()
                            to_mask = (abs_delta >= thresh) & (~self._mask[n])
                            self._mask[n] = to_mask | self._mask[n]
                            n_masked += to_mask.sum()
                            logger.info(f'Setting masks for layer {n}')      
                    logger.info(f'Unmasked {n_masked} params in {pattern_str}')       


        def train(self, **kwargs):
            self.freeze()
            result = None
            
            for it in range(self.sft_args.n_ft_iterations):
                logger.info(f'Fine-tuning iteration {it+1}')
                with torch.no_grad():
                    previous_params = {
                        n: torch.zeros_like(p, device='cpu').copy_(p)
                        for n, p in self.model.named_parameters()
                    }

                self.disable_masking()
                self.optimizer = None
                self.lr_scheduler = None
                self.set_training_len(
                    self.sft_args.full_ft_min_steps_per_iteration,
                    self.sft_args.full_ft_max_steps_per_iteration,
                    self.sft_args.full_ft_max_epochs_per_iteration,
                )
                super().train(**kwargs)

                if self.unfreeze_strategy == 'global':
                    self.unfreeze_k_most_changed_params(
                        self.n_tunable_params // self.sft_args.n_ft_iterations
                    )
                elif self.unfreeze_strategy == 'layer_wise_selection':
                    self.unfreeze_k_most_changed_params_per_layer_percentage()
                else:
                    raise ValueError(f"Unknown unfreeze_strategy: {self.unfreeze_strategy}")
                                
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.copy_(previous_params[n])

                self.enable_masking()
                self.optimizer = None
                self.lr_scheduler = None
                self.set_training_len(
                    self.sft_args.sparse_ft_min_steps_per_iteration,
                    self.sft_args.sparse_ft_max_steps_per_iteration,
                    self.sft_args.sparse_ft_max_epochs_per_iteration,
                )
                result = super().train(**kwargs)
            
            return result

    return _LotteryTicketSparseFineTuner
