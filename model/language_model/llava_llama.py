#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class Router(nn.Module):
    def __init__(self, embed_dim=4096, num_heads=16, mlp_ratio=2.0, gamma=0.6, noise_std=0.07, temperature=0.05):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        
        self.gamma = gamma
        self.noise_std = noise_std
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=0)

    def forward(self, vision_feature, text_embed, attention_mask):
        # Assuming vision_feature has shape (L_v, D)
        # text_embed has shape (L_t, D)
        # attention_mask has shape (L_t,)
        
        L_v, _ = vision_feature.shape
        L_t, _ = text_embed.shape
        
        # Cross attention layer
        query = vision_feature.unsqueeze(1)  # (L_v, 1, D)
        key = value = text_embed.unsqueeze(0)  # (1, L_t, D)
        
        # Derive attention mask for vision feature
        vision_attention_mask = attention_mask.unsqueeze(0).repeat(self.cross_attn.num_heads, L_v, 1)  # (num_heads, L_v, L_t)
        
        # Cross attention
        attn_output, _ = self.cross_attn(query, key, value, attn_mask=vision_attention_mask)
        attn_output = attn_output.squeeze(1)  # (L_v, D)
        
        # Residual connection and layer normalization
        attn_output = self.norm(vision_feature + attn_output)
        
        # MLP for generating weight scores
        weight_scores = self.mlp(attn_output).squeeze(-1)  # (L_v,)
        
        # Add Gaussian noise to scores for diversity
        noise = torch.randn_like(weight_scores) * self.noise_std
        weight_scores = weight_scores + noise
        
        # Apply temperature-scaled softmax to scores
        scores = self.softmax(weight_scores / self.temperature)
        
        # Sort scores and get indices
        sorted_scores, indices = torch.sort(scores, descending=True)
        
        # Calculate cumulative sum of sorted scores
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        
        # Find the threshold index
        threshold_index = (cum_scores <= self.gamma).sum()
        #print(threshold_index)
        # Select vision tokens up to the threshold index
        selected_indices = indices[:threshold_index]
        selected_vision_feature = vision_feature[selected_indices]
        
        return selected_vision_feature


class CosSimRouter(nn.Module):
    def __init__(self, model=None, gamma=0.2, noise_std=0.00, temperature=0.05, top_k=0):
        super().__init__()
        self.gamma = gamma
        self.noise_std = noise_std
        self.temperature = temperature
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=0)
        # Extract the weights from the specified layers
        self.q_proj_weight = None
        self.k_proj_weight = None
        if model != None:
           
            for name, module in model.named_modules():
                if name == 'layers.0.self_attn.q_proj':
                    self.q_proj_weight = module.weight
                elif name == 'layers.0.self_attn.k_proj':
                    self.k_proj_weight = module.weight
            
            assert self.q_proj_weight is not None and self.k_proj_weight is not None, "Specified layers not found in the model"
        
    
    def forward(self, vision_feature, text_embed, attention_mask):
        # Assuming vision_feature has shape (L_v, D)
        # text_embed has shape (L_t, D)
        # attention_mask has shape (L_t,)
        
        L_v, D = vision_feature.shape
        L_t, D = text_embed.shape
        if self.q_proj_weight != None:
        # Apply the extracted weights to the vision feature and text embedding
            vision_query = F.linear(vision_feature, self.q_proj_weight)  # (L_v, D)
            text_key = F.linear(text_embed, self.k_proj_weight)  # (L_t, D)
        
            # Calculate cosine similarity matrix
            cos_sim_matrix = F.cosine_similarity(vision_query.unsqueeze(1), text_key.unsqueeze(0), dim=-1)  # (L_v, L_t)
        else:
            cos_sim_matrix = F.cosine_similarity(vision_feature.unsqueeze(1), text_embed.unsqueeze(0), dim=-1)  # (L_v, L_t) 
        # Apply attention mask to cosine similarity matrix
        cos_sim_matrix = cos_sim_matrix.masked_fill(attention_mask == False, 0.0)
        
        # Get the maximum value along the text (L_t) axis for each vision token
        scores, _ = cos_sim_matrix.max(dim=-1)  # (L_v,)
        
        # Add Gaussian noise to scores for diversity
        noise = torch.randn_like(scores) * self.noise_std
        scores = scores + noise
        
        # Apply temperature-scaled softmax to scores
        scores = self.softmax(scores / self.temperature)
        
        # Sort scores and get indices
        sorted_scores, indices = torch.sort(scores, descending=True)
        
        # Calculate cumulative sum of sorted scores
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        
        # Find the threshold index
        threshold_index = (cum_scores <= self.gamma).sum()
       
        # Select vision tokens up to the threshold index
        selected_indices = indices[:threshold_index]
        print(len(selected_indices))
        selected_vision_feature = vision_feature[selected_indices]
        if self.top_k == 0:
            return selected_vision_feature
        
        else:

            # Calculate cosine similarity between selected vision tokens and all vision tokens
            selected_cos_sim_matrix = F.cosine_similarity(selected_vision_feature.unsqueeze(1), vision_feature.unsqueeze(0), dim=-1)  # (threshold_index, L_v)
            
            # Get top-k similar vision tokens for each selected token
            _, top_k_indices = selected_cos_sim_matrix.topk(self.top_k, dim=-1)  # (threshold_index, top_k)
            
            # Gather top-k similar vision tokens
            top_k_vision_feature = vision_feature[top_k_indices]  # (threshold_index, top_k, D)
            
            # Calculate softmax weights for top-k similar tokens
            top_k_cos_sim = selected_cos_sim_matrix.gather(1, top_k_indices)  # (threshold_index, top_k)
            top_k_weights = F.softmax(top_k_cos_sim, dim=-1)  # (threshold_index, top_k)
            
            # Compute weighted sum of top-k similar tokens
            weighted_sum = torch.sum(top_k_vision_feature * top_k_weights.unsqueeze(-1), dim=1)  # (threshold_index, D)
            
            return weighted_sum

class CosSimRouter_pad(nn.Module):
    def __init__(self, model=None, gamma=0.07, noise_std=0.00, temperature=0.05, top_k=0, padding_size=1):
        super().__init__()
        self.gamma = gamma
        self.noise_std = noise_std
        self.temperature = temperature
        self.top_k = top_k
        self.padding_size = padding_size
        self.softmax = nn.Softmax(dim=0)
        # Extract the weights from the specified layers
        self.q_proj_weight = None
        self.k_proj_weight = None
        if model != None:
           
            for name, module in model.named_modules():
                if name == 'layers.0.self_attn.q_proj':
                    self.q_proj_weight = module.weight
                elif name == 'layers.0.self_attn.k_proj':
                    self.k_proj_weight = module.weight
            
            assert self.q_proj_weight is not None and self.k_proj_weight is not None, "Specified layers not found in the model"
        
    
    def forward(self, vision_feature, text_embed, attention_mask):
        # Assuming vision_feature has shape (L_v, D)
        # text_embed has shape (L_t, D)
        # attention_mask has shape (L_t,)
        
        L_v, D = vision_feature.shape
        L_t, D = text_embed.shape
        if self.q_proj_weight != None:
        # Apply the extracted weights to the vision feature and text embedding
            vision_query = F.linear(vision_feature, self.q_proj_weight)  # (L_v, D)
            text_key = F.linear(text_embed, self.k_proj_weight)  # (L_t, D)
        
            # Calculate cosine similarity matrix
            cos_sim_matrix = F.cosine_similarity(vision_query.unsqueeze(1), text_key.unsqueeze(0), dim=-1)  # (L_v, L_t)
        else:
            cos_sim_matrix = F.cosine_similarity(vision_feature.unsqueeze(1), text_embed.unsqueeze(0), dim=-1)  # (L_v, L_t) 
        # Apply attention mask to cosine similarity matrix
        cos_sim_matrix = cos_sim_matrix.masked_fill(attention_mask == False, 0.0)
        
        # Get the maximum value along the text (L_t) axis for each vision token
        scores, _ = cos_sim_matrix.max(dim=-1)  # (L_v,)
        
        # Add Gaussian noise to scores for diversity
        noise = torch.randn_like(scores) * self.noise_std
        scores = scores + noise
        
        # Apply temperature-scaled softmax to scores
        scores = self.softmax(scores / self.temperature)
        
        # Sort scores and get indices
        sorted_scores, indices = torch.sort(scores, descending=True)
        
        # Calculate cumulative sum of sorted scores
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        
        # Find the threshold index
        threshold_index = (cum_scores <= self.gamma).sum()
       
        # Select vision tokens up to the threshold index
        selected_indices = indices[:threshold_index]
        selected_vision_feature = vision_feature[selected_indices]
        
        if self.padding_size > 0:
            # Get the neighboring indices of the selected tokens
            selected_indices_grid = self.get_neighbor_indices(selected_indices, 24, self.padding_size)
            
            # Flatten the selected indices and neighbor indices
            selected_indices_flat = selected_indices_grid.flatten()
            
            # Remove duplicate indices
            unique_indices = torch.unique(selected_indices_flat)
            
            # Gather the selected and neighboring vision tokens
            selected_vision_feature = vision_feature[unique_indices]
            
        
        if self.top_k == 0:
            return selected_vision_feature
        
        else:
            # Calculate cosine similarity between selected vision tokens and all vision tokens
            selected_cos_sim_matrix = F.cosine_similarity(selected_vision_feature.unsqueeze(1), vision_feature.unsqueeze(0), dim=-1)  # (threshold_index, L_v)
            
            # Get top-k similar vision tokens for each selected token
            _, top_k_indices = selected_cos_sim_matrix.topk(self.top_k, dim=-1)  # (threshold_index, top_k)
            
            # Gather top-k similar vision tokens
            top_k_vision_feature = vision_feature[top_k_indices]  # (threshold_index, top_k, D)
            
            # Calculate softmax weights for top-k similar tokens
            top_k_cos_sim = selected_cos_sim_matrix.gather(1, top_k_indices)  # (threshold_index, top_k)
            top_k_weights = F.softmax(top_k_cos_sim, dim=-1)  # (threshold_index, top_k)
            
            # Compute weighted sum of top-k similar tokens
            weighted_sum = torch.sum(top_k_vision_feature * top_k_weights.unsqueeze(-1), dim=1)  # (threshold_index, D)
            
            return weighted_sum
    
    def get_neighbor_indices(self, indices, grid_size, padding_size):
        # Convert indices to grid coordinates
        grid_coords = torch.stack((indices // grid_size, indices % grid_size), dim=1)

        # Generate offsets for neighboring tokens based on padding size
        offsets = []
        for i in range(-padding_size, padding_size + 1):
            for j in range(-padding_size, padding_size + 1):
                if i == 0 and j == 0:
                    continue
                offsets.append([i, j])
        offsets = torch.tensor(offsets).to(indices.device)

        # Apply padding to the grid coordinates
        padded_coords = grid_coords.unsqueeze(1) + offsets.unsqueeze(0).repeat(grid_coords.size(0), 1, 1)
        padded_coords = torch.clamp(padded_coords, min=0, max=grid_size - 1)

        # Convert padded coordinates back to indices
        padded_indices = padded_coords[:, :, 0] * grid_size + padded_coords[:, :, 1]

        return padded_indices

class CosSimRouter_expand(nn.Module):
    def __init__(self, model = None, initial_gamma=0.3, expand_gamma=0.1, noise_std=0.00, temperature=0.1, expand_temp=0.01):
        super().__init__()
        self.initial_gamma = initial_gamma
        self.expand_gamma = expand_gamma
        self.noise_std = noise_std
        self.temperature = temperature
        self.expand_temp = expand_temp
        self.softmax = nn.Softmax(dim=0)
        self.q_proj_weight = None
        self.k_proj_weight = None
        if model != None:
           
            for name, module in model.named_modules():
                if name == 'layers.0.self_attn.q_proj':
                    self.q_proj_weight = module.weight
                elif name == 'layers.0.self_attn.k_proj':
                    self.k_proj_weight = module.weight
            
            assert self.q_proj_weight is not None and self.k_proj_weight is not None, "Specified layers not found in the model"

    def forward(self, vision_feature, text_embed, attention_mask):
        device = vision_feature.device
        L_v, D = vision_feature.shape
        L_t, D = text_embed.shape

        # Initial selection
        if self.k_proj_weight != None:
        # Apply the extracted weights to the vision feature and text embedding
            vision_key = F.linear(vision_feature, self.k_proj_weight)  # (L_v, D)
            text_key = F.linear(text_embed, self.k_proj_weight)  # (L_t, D)
            cos_sim_matrix = F.cosine_similarity(vision_key.unsqueeze(1), text_key.unsqueeze(0), dim=-1)
        else:
            cos_sim_matrix = F.cosine_similarity(vision_feature.unsqueeze(1), text_embed.unsqueeze(0), dim=-1)
        cos_sim_matrix = cos_sim_matrix.masked_fill(attention_mask == False, 0.0)
        scores, _ = cos_sim_matrix.max(dim=-1)
        noise = torch.randn_like(scores, device=device) * self.noise_std
        scores = scores + noise
        scores = self.softmax(scores / self.temperature)
        expand_scores = self.softmax(scores/self.expand_temp)
        sorted_scores, indices = torch.sort(scores, descending=True)
        
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        threshold_index = (cum_scores <= self.initial_gamma).sum()
        selected_indices = indices[:threshold_index]
        selected_vision_feature = vision_feature[selected_indices]
        selected_scores = scores[selected_indices]
        expand_selected_scores = expand_scores[selected_indices]
        
        # Expansion
        if self.k_proj_weight != None:
        # Apply the extracted weights to the vision feature and text embedding
            vision_key = F.linear(vision_feature, self.k_proj_weight)  # (L_v, D)
            selected_vision_key = F.linear(selected_vision_feature, self.k_proj_weight)  # (L_t, D)
            cos_sim_expand = F.cosine_similarity(selected_vision_key.unsqueeze(1), vision_key.unsqueeze(0), dim=-1)
            weighted_cos_sim = expand_selected_scores.unsqueeze(1) * cos_sim_expand
        else:
            cos_sim_expand = F.cosine_similarity(selected_vision_feature.unsqueeze(1), vision_feature.unsqueeze(0), dim=-1)
            weighted_cos_sim = expand_selected_scores.unsqueeze(1) * cos_sim_expand
        softmax_weights = F.softmax(weighted_cos_sim.view(-1), dim=0).view_as(weighted_cos_sim)
        aggregated_scores = softmax_weights.sum(dim=0)

        # Select expanded patches
        sorted_expand_scores, expand_indices = torch.sort(aggregated_scores, descending=True)
        cum_expand_scores = torch.cumsum(sorted_expand_scores, dim=0)
        expand_threshold_index = (cum_expand_scores <= self.expand_gamma).sum()
        expanded_indices = expand_indices[:expand_threshold_index]

        # Remove overlap with initial selection
        expanded_indices = expanded_indices[~torch.isin(expanded_indices, selected_indices)]
        
        # Combine initial and expanded indices
        final_indices = torch.cat([selected_indices, expanded_indices])
        final_vision_feature = vision_feature[final_indices]
        
        return final_vision_feature

class CosSimRouter_learn(nn.Module):
    def __init__(self, initial_gamma=0.2, expand_ratio=0.7, noise_std=0.00, temp=0.05, initial_temperature=1.0, final_temperature=0.1, temperature_anneal_steps=30000, embed_dim=4096):
        super().__init__()
        self.initial_gamma = initial_gamma
        self.expand_ratio = expand_ratio
        self.noise_std = noise_std
        self.temp = temp
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_anneal_steps = temperature_anneal_steps
        self.step = 0
        self.softmax = nn.Softmax(dim=0)
        self.expander = ExpanderModule(embed_dim=embed_dim)
 
    def gumbel_softmax_sample(self, logits, tau=1, hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim=-1)
        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def get_temperature(self):
        if self.step < self.temperature_anneal_steps:
            tau = self.initial_temperature - (self.initial_temperature - self.final_temperature) * (self.step / self.temperature_anneal_steps)
        else:
            tau = self.final_temperature
        return tau

    def forward(self, vision_feature, text_embed, attention_mask, inference_mode=False):
        device = vision_feature.device
        L_v, D = vision_feature.shape
        L_t, D = text_embed.shape

        # Initial selection
        cos_sim_matrix = F.cosine_similarity(vision_feature.unsqueeze(1), text_embed.unsqueeze(0), dim=-1)
        cos_sim_matrix = cos_sim_matrix.masked_fill(attention_mask == False, 0.0)
        scores, _ = cos_sim_matrix.max(dim=-1)
        noise = torch.randn_like(scores, device=device) * self.noise_std
        scores = scores + noise
        scores = self.softmax(scores / self.temp)
        sorted_scores, indices = torch.sort(scores, descending=True)
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        threshold_index = (cum_scores <= self.initial_gamma).sum()
        selected_indices = indices[:threshold_index]
        selected_vision_feature = vision_feature[selected_indices]

        # Separate remained patches
        remained_indices = indices[threshold_index:]
        remained_vision_feature = vision_feature[remained_indices]
        
        # Expansion
        #self.expander.half()
        expand_scores = self.expander(selected_vision_feature, remained_vision_feature, text_embed)
        
        if inference_mode:
            # Hard selection during inference
            _, expand_indices = torch.topk(expand_scores, k=int(threshold_index * self.expand_ratio))
        else:
            # Use Gumbel-Softmax probabilities directly for expansion
            gumbel_expand_scores = self.gumbel_softmax_sample(expand_scores, tau=self.get_temperature(), hard=False)
            num_expand = int(threshold_index * self.expand_ratio)
            expand_indices = gumbel_expand_scores.multinomial(num_samples=num_expand, replacement=False)
        
        # Convert expand_indices from remained to original indices
        expand_indices = remained_indices[expand_indices]
        
        # Combine initial and expanded indices
        final_indices = torch.cat([selected_indices, expand_indices])
        final_indices, _ = torch.sort(final_indices)  # Sort to maintain order
        
        final_vision_feature = vision_feature[final_indices]
        
        self.step += 1  # Update step for annealing

        return final_vision_feature



class CosSimRouter_learnable_pad(nn.Module):
    def __init__(self, initial_gamma=0.065, expand_ratio=0.3, noise_std=0.00, temp=0.05, initial_temperature=1.0, final_temperature=0.1, temperature_anneal_steps=30000, embed_dim=4096, grid_size=24, padding_size=1, local_expand_num=8):
        super().__init__()
        self.initial_gamma = initial_gamma
        self.expand_ratio = expand_ratio
        self.noise_std = noise_std
        self.temp = temp
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_anneal_steps = temperature_anneal_steps
        self.step = 0
        self.softmax = nn.Softmax(dim=0)
        self.global_expander = ExpanderModule(embed_dim=embed_dim)
        self.grid_size = grid_size
        self.padding_size = padding_size
        self.local_expand_num = local_expand_num
        self.to("cuda:0")
        self.half()
 
    def gumbel_softmax_sample(self, logits, tau=1, hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim=-1)
        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def get_temperature(self):
        if self.step < self.temperature_anneal_steps:
            tau = self.initial_temperature - (self.initial_temperature - self.final_temperature) * (self.step / self.temperature_anneal_steps)
        else:
            tau = self.final_temperature
        return tau

    def forward(self, vision_feature, text_embed, attention_mask, inference_mode=False):
        device = vision_feature.device
        L_v, D = vision_feature.shape
        L_t, D = text_embed.shape

        # Initial selection
        cos_sim_matrix = F.cosine_similarity(vision_feature.unsqueeze(1), text_embed.unsqueeze(0), dim=-1)
        cos_sim_matrix = cos_sim_matrix.masked_fill(attention_mask == False, 0.0)
        scores, _ = cos_sim_matrix.max(dim=-1)
        noise = torch.randn_like(scores, device=device) * self.noise_std
        scores = scores + noise
        scores = self.softmax(scores / self.temp)
        sorted_scores, indices = torch.sort(scores, descending=True)
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        threshold_index = (cum_scores <= self.initial_gamma).sum()
        selected_indices = indices[:threshold_index]
        selected_vision_feature = vision_feature[selected_indices]
        selected_indices_grid = self.get_neighbor_indices(selected_indices, 24, self.padding_size)
            
        # Flatten the selected indices and neighbor indices
        selected_indices_flat = selected_indices_grid.flatten()
        
        # Remove duplicate indices
        unique_indices = torch.unique(selected_indices_flat)
        
        # Gather the selected and neighboring vision tokens
        selected_vision_feature = vision_feature[unique_indices]
        # Local Expansion
        '''
        local_selected_indices = []

        for idx in selected_indices:
            # Get the neighboring indices of the selected token
            neighbor_indices_grid = self.get_neighbor_indices(torch.tensor([idx], device=device), self.grid_size, self.padding_size)
            neighbor_indices_flat = neighbor_indices_grid.flatten()
            
            # Remove duplicates and the selected index itself
            unique_neighbor_indices = torch.unique(neighbor_indices_flat)
            neighbor_indices = unique_neighbor_indices[unique_neighbor_indices != idx]
            
            # Gather neighbor patches
            neighbor_patches = vision_feature[neighbor_indices]
            
            # Perform local expansion based on cosine similarity
            local_cos_sim = F.cosine_similarity(vision_feature[idx].unsqueeze(0), neighbor_patches, dim=-1)
            
            # Directly apply top-k on cosine similarity
            _, local_expand_indices = torch.topk(local_cos_sim, k=self.local_expand_num)
            
            # Convert local_expand_indices from neighbor to original indices
            local_expand_indices = neighbor_indices[local_expand_indices]
            local_selected_indices.append(torch.cat([torch.tensor([idx], device=device), local_expand_indices]))
        '''
        # Combine all local selected indices
        #local_selected_indices = torch.cat(local_selected_indices)
        
        # Remove duplicates and keep the original order
        #local_selected_indices = torch.unique(local_selected_indices)
        
        # Global Expansion
        remained_indices = torch.tensor([i for i in range(L_v) if i not in unique_indices], device=device)
        remained_vision_feature = vision_feature[remained_indices]

        global_expand_scores = self.global_expander(selected_vision_feature, remained_vision_feature, text_embed)
        
        if inference_mode:
            # Hard selection during inference
            _, global_expand_indices = torch.topk(global_expand_scores, k=int(threshold_index * self.expand_ratio))
        else:
            # Use Gumbel-Softmax probabilities directly for expansion
            gumbel_expand_scores = self.gumbel_softmax_sample(global_expand_scores, tau=self.get_temperature(), hard=False)
            num_expand = int(len(unique_indices) * self.expand_ratio)
            global_expand_indices = gumbel_expand_scores.multinomial(num_samples=num_expand, replacement=False)
        
        # Convert global_expand_indices from remained to original indices
        global_expand_indices = remained_indices[global_expand_indices]
        
        # Combine local and global indices
        final_indices = torch.cat([unique_indices, global_expand_indices])
        #final_indices = local_selected_indices
        # Remove duplicates and keep the original order
        final_indices = torch.unique(final_indices)
        
        final_vision_feature = vision_feature[final_indices]
        
        self.step += 1  # Update step for annealing
        
        return final_vision_feature

    def get_neighbor_indices(self, indices, grid_size, padding_size):
        # Convert indices to grid coordinates
        grid_coords = torch.stack((indices // grid_size, indices % grid_size), dim=1)

        # Generate offsets for neighboring tokens based on padding size
        offsets = []
        for i in range(-padding_size, padding_size + 1):
            for j in range(-padding_size, padding_size + 1):
                if i == 0 and j == 0:
                    continue
                offsets.append([i, j])
        offsets = torch.tensor(offsets).to(indices.device)

        # Apply padding to the grid coordinates
        padded_coords = grid_coords.unsqueeze(1) + offsets.unsqueeze(0).repeat(grid_coords.size(0), 1, 1)
        padded_coords = torch.clamp(padded_coords, min=0, max=grid_size - 1)

        # Convert padded coordinates back to indices
        padded_indices = padded_coords[:, :, 0] * grid_size + padded_coords[:, :, 1]

        return padded_indices

class TokenSelectionModule(nn.Module):
    def __init__(self, embed_dim=4096, num_heads=8, hidden_dim=2048, dropout=0.1, selection_ratio=0.5):
        super().__init__()
        
        self.selection_ratio = selection_ratio  # Ratio of tokens to select during inference
        
        # Cross-attention layers
        self.global_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.local_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # MLP for generating probabilities
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: (L_v, 2, 1)
        )
    
    def forward(self, vision_feature, text_embed, inference_mode=False):
        L_v, D = vision_feature.shape
        L_t, _ = text_embed.shape
        
        # Concatenate vision features and text embeddings
        combined_features = torch.cat([vision_feature, text_embed], dim=0)  # (L_v + L_t, D)
        
        # Global features: vision tokens as query, combined features as key and value
        global_features, _ = self.global_cross_attn(vision_feature, combined_features, combined_features)  # (L_v, D)
        
        # Local features: vision tokens as query, text embeddings as key and value
        local_features, _ = self.local_cross_attn(vision_feature, text_embed, text_embed)  # (L_v, D)
        local_features = local_features + vision_feature
        # Stack local and global features to get the output shape (L_v, D, 2)
        combined_local_global = torch.stack([local_features, global_features], dim=-1)  # (L_v, D, 2)
        
        # Transpose to get the shape (L_v, 2, D)
        combined_local_global = combined_local_global.transpose(1, 2)  # (L_v, 2, D)
        
        # MLP to generate logits for keeping or dropping tokens, applied along the final dimension
        logits = self.mlp(combined_local_global).squeeze(-1)  # (L_v, 2)
        second_linear_weight = self.mlp[2].weight  # Weight of the second linear layer
        second_linear_bias = self.mlp[2].bias  # Bias of the second linear layer
        
        # Print the parameters
        #print("Second Linear Layer Weight:", second_linear_weight)
        #print("Second Linear Layer Bias:", second_linear_bias)
        if inference_mode:
            # Select tokens based on highest logits during inference
            logits_max = logits[:, 1]  # Use the second column of logits (corresponding to "keep")
            num_select = int(self.selection_ratio * L_v)
            _, selected_indices = torch.topk(logits_max, num_select, dim=0)
            
            pruned_vision_feature = vision_feature[selected_indices]
        else:
            # Apply Gumbel Softmax along the last dimension to get the decision
            gumbel_output = F.gumbel_softmax(logits, tau=1, hard=True, dim=-1)  # (L_v, 2)
            
            # The second column of Gumbel Softmax output is the decision mask
            decision_mask = gumbel_output[:, 1]  # (L_v,)
            
            # Multiply the decision mask with vision features to select tokens
            selected_vision_feature = vision_feature * decision_mask.unsqueeze(-1)  # (L_v, D)
            
            # Output only the vision features where the decision mask is 1
            selected_indices = decision_mask.bool().nonzero(as_tuple=True)[0]  # (L_selected,)
            pruned_vision_feature = selected_vision_feature[selected_indices]  # (L_selected, D)
            
        return pruned_vision_feature, selected_indices

class StackedTokenSelectionModule(nn.Module):
    def __init__(self, embed_dim=4096, num_heads=8, hidden_dim=2048, dropout=0.1, selection_ratio=0.5):
        super().__init__()
        self.module1 = TokenSelectionModule(embed_dim, num_heads, hidden_dim, dropout, selection_ratio)
        self.module2 = TokenSelectionModule(embed_dim, num_heads, hidden_dim, dropout, selection_ratio)
    
    def forward(self, vision_feature, text_embed, inference_mode=False):
        # First round of selection
        selected_vision_feature_1, selected_indices_1 = self.module1(vision_feature, text_embed, inference_mode)
        
        # Second round of selection
        selected_vision_feature_2, selected_indices_2 = self.module2(selected_vision_feature_1, text_embed, inference_mode)
        
        # Combine indices to get the final indices relative to the original input
        final_selected_indices = selected_indices_1[selected_indices_2]
        
        return selected_vision_feature_2, final_selected_indices
    
class CosSimRouter_pad_merge_entropy(nn.Module):
    def __init__(self, model=None, gamma=0.07, noise_std=0.00, temperature=0.05, top_k=0, padding_size=1):
        super().__init__()
        self.gamma = gamma
        self.noise_std = noise_std
        self.temperature = temperature
        self.top_k = top_k
        self.padding_size = padding_size
        self.softmax = nn.Softmax(dim=-1)
        # Extract the weights from the specified layers
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.selector = StackedTokenSelectionModule()
        if model is not None:
            for name, module in model.named_modules():
                if name == 'layers.0.self_attn.q_proj':
                    self.q_proj_weight = module.weight
                elif name == 'layers.0.self_attn.k_proj':
                    self.k_proj_weight = module.weight
            assert self.q_proj_weight is not None and self.k_proj_weight is not None, "Specified layers not found in the model"
    
   
    def forward(self, vision_feature, text_embed, attention_mask):
        L_v, D = vision_feature.shape
        L_t, D = text_embed.shape
        selected_vision_feature, selected_indices = self.selector(vision_feature, text_embed)
        
        
        remaining_indices = torch.arange(L_v, device=vision_feature.device)[~torch.isin(torch.arange(L_v, device=vision_feature.device), selected_indices)]
        remaining_vision_feature = vision_feature[remaining_indices]
        
        if remaining_vision_feature.size(0) > 0:
            cos_sim_remain_selected = F.cosine_similarity(remaining_vision_feature.unsqueeze(1), selected_vision_feature.unsqueeze(0), dim=-1)  # (L_r, L_s)
            best_match_indices = torch.argmax(cos_sim_remain_selected, dim=1)  # (L_r,)
            for i in range(selected_vision_feature.size(0)):
                matching_remain_indices = (best_match_indices == i).nonzero(as_tuple=True)[0]
                if matching_remain_indices.size(0) > 0:
                    merging_tokens = torch.cat([selected_vision_feature[i].unsqueeze(0), remaining_vision_feature[matching_remain_indices]], dim=0)  # (K, D)
                    token_mean = merging_tokens.mean(dim=0)  # (D,)
                    norms = torch.norm(merging_tokens, dim=1, keepdim=True)  # (K, 1)
                    token_mean = merging_tokens.mean(dim=0)  # (D,)
                    norm = torch.norm(token_mean)
                    mean_normalized = token_mean/norm  # (D,)
                    max_norm = norms.max()  # scalar
                    updated_patch = mean_normalized * max_norm  # (D,)
                    selected_vision_feature[i] = updated_patch
        
        if self.top_k == 0:
            return selected_vision_feature
        
        else:
            selected_cos_sim_matrix = F.cosine_similarity(selected_vision_feature.unsqueeze(1), vision_feature.unsqueeze(0), dim=-1)  # (threshold_index, L_v)
            _, top_k_indices = selected_cos_sim_matrix.topk(self.top_k, dim=-1)  # (threshold_index, top_k)
            top_k_vision_feature = vision_feature[top_k_indices]  # (threshold_index, top_k, D)
            top_k_cos_sim = selected_cos_sim_matrix.gather(1, top_k_indices)  # (threshold_index, top_k)
            top_k_weights = F.softmax(top_k_cos_sim, dim=-1)  # (threshold_index, top_k)
            weighted_sum = torch.sum(top_k_vision_feature * top_k_weights.unsqueeze(-1), dim=1)  # (threshold_index, D)
            return weighted_sum
    
    def get_neighbor_indices(self, indices, grid_size, padding_size):
        grid_coords = torch.stack((indices // grid_size, indices % grid_size), dim=1)
        offsets = []
        for i in range(-padding_size, padding_size + 1):
            for j in range(-padding_size, padding_size + 1):
                if i == 0 and j == 0:
                    continue
                offsets.append([i, j])
        offsets = torch.tensor(offsets).to(indices.device)
        padded_coords = grid_coords.unsqueeze(1) + offsets.unsqueeze(0).repeat(grid_coords.size(0), 1, 1)
        padded_coords = torch.clamp(padded_coords, min=0, max=grid_size - 1)
        padded_indices = padded_coords[:, :, 0] * grid_size + padded_coords[:, :, 1]
        return padded_indices
    
class CosSimRouter_pad_merge(nn.Module):
    def __init__(self, model=None, gamma=0.05, noise_std=0.00, temperature=0.05, top_k=0, padding_size=1):
        super().__init__()
        self.gamma = gamma
        self.noise_std = noise_std
        self.temperature = temperature
        self.top_k = top_k
        self.padding_size = padding_size
        self.softmax = nn.Softmax(dim=0)
        # Extract the weights from the specified layers
        self.q_proj_weight = None
        self.k_proj_weight = None
        if model is not None:
            for name, module in model.named_modules():
                if name == 'layers.0.self_attn.q_proj':
                    self.q_proj_weight = module.weight
                elif name == 'layers.0.self_attn.k_proj':
                    self.k_proj_weight = module.weight
            assert self.q_proj_weight is not None and self.k_proj_weight is not None, "Specified layers not found in the model"
        
    
    def forward(self, vision_feature, text_embed, attention_mask):
        L_v, D = vision_feature.shape
        L_t, D = text_embed.shape
        if self.q_proj_weight is not None:
            # Apply the extracted weights to the vision feature and text embedding
            vision_query = F.linear(vision_feature, self.q_proj_weight)  # (L_v, D)
            text_key = F.linear(text_embed, self.k_proj_weight)  # (L_t, D)
            # Calculate cosine similarity matrix
            cos_sim_matrix = F.cosine_similarity(vision_query.unsqueeze(1), text_key.unsqueeze(0), dim=-1)  # (L_v, L_t)
        else:
            cos_sim_matrix = F.cosine_similarity(vision_feature.unsqueeze(1), text_embed.unsqueeze(0), dim=-1)  # (L_v, L_t)
        
        # Apply attention mask to cosine similarity matrix
        cos_sim_matrix = cos_sim_matrix.masked_fill(attention_mask == False, 0.0)
        
        # Get the maximum value along the text (L_t) axis for each vision token
        scores, _ = cos_sim_matrix.max(dim=-1)  # (L_v,)
        
        # Add Gaussian noise to scores for diversity
        noise = torch.randn_like(scores) * self.noise_std
        scores = scores + noise
        
        # Apply temperature-scaled softmax to scores
        scores = self.softmax(scores / self.temperature)
        
        # Sort scores and get indices
        sorted_scores, indices = torch.sort(scores, descending=True)
        
        # Calculate cumulative sum of sorted scores
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        
        # Find the threshold index
        threshold_index = (cum_scores <= self.gamma).sum()
       
        # Select vision tokens up to the threshold index
        selected_indices = indices[:threshold_index]
        selected_vision_feature = vision_feature[selected_indices]
        
        if self.padding_size > 0:
            # Get the neighboring indices of the selected tokens
            selected_indices_grid = self.get_neighbor_indices(selected_indices, int(L_v ** 0.5), self.padding_size)
            
            # Flatten the selected indices and neighbor indices
            selected_indices_flat = selected_indices_grid.flatten()
            
            # Remove duplicate indices
            unique_indices = torch.unique(selected_indices_flat)
            
            # Gather the selected and neighboring vision tokens
            selected_vision_feature = vision_feature[unique_indices]
            selected_indices = unique_indices  # Update selected_indices to include neighbors
        
        # Merge remaining patches with selected patches
        remaining_indices = torch.arange(L_v, device=vision_feature.device)[~torch.isin(torch.arange(L_v, device=vision_feature.device), selected_indices)]
        remaining_vision_feature = vision_feature[remaining_indices]
        
        if remaining_vision_feature.size(0) > 0:
            # Calculate cosine similarity between remaining and selected (with neighbors) patches
            cos_sim_remain_selected = F.cosine_similarity(remaining_vision_feature.unsqueeze(1), selected_vision_feature.unsqueeze(0), dim=-1)  # (L_r, L_s)
            
            # Find the selected patch with the highest cosine similarity for each remaining patch
            best_match_indices = torch.argmax(cos_sim_remain_selected, dim=1)  # (L_r,)
            
            # Merge remaining patches into selected patches by averaging
            for i in range(selected_vision_feature.size(0)):
                matching_remain_indices = (best_match_indices == i).nonzero(as_tuple=True)[0]
                if matching_remain_indices.size(0) > 0:
                    selected_vision_feature[i] = torch.mean(torch.cat([selected_vision_feature[i].unsqueeze(0), remaining_vision_feature[matching_remain_indices]], dim=0), dim=0)
        
        if self.top_k == 0:
            return selected_vision_feature
        
        else:
            # Calculate cosine similarity between selected vision tokens and all vision tokens
            selected_cos_sim_matrix = F.cosine_similarity(selected_vision_feature.unsqueeze(1), vision_feature.unsqueeze(0), dim=-1)  # (threshold_index, L_v)
            
            # Get top-k similar vision tokens for each selected token
            _, top_k_indices = selected_cos_sim_matrix.topk(self.top_k, dim=-1)  # (threshold_index, top_k)
            
            # Gather top-k similar vision tokens
            top_k_vision_feature = vision_feature[top_k_indices]  # (threshold_index, top_k, D)
            
            # Calculate softmax weights for top-k similar tokens
            top_k_cos_sim = selected_cos_sim_matrix.gather(1, top_k_indices)  # (threshold_index, top_k)
            top_k_weights = F.softmax(top_k_cos_sim, dim=-1)  # (threshold_index, top_k)
            
            # Compute weighted sum of top-k similar tokens
            weighted_sum = torch.sum(top_k_vision_feature * top_k_weights.unsqueeze(-1), dim=1)  # (threshold_index, D)
            
            return weighted_sum
    
    def get_neighbor_indices(self, indices, grid_size, padding_size):
        # Convert indices to grid coordinates
        grid_coords = torch.stack((indices // grid_size, indices % grid_size), dim=1)

        # Generate offsets for neighboring tokens based on padding size
        offsets = []
        for i in range(-padding_size, padding_size + 1):
            for j in range(-padding_size, padding_size + 1):
                if i == 0 and j == 0:
                    continue
                offsets.append([i, j])
        offsets = torch.tensor(offsets).to(indices.device)

        # Apply padding to the grid coordinates
        padded_coords = grid_coords.unsqueeze(1) + offsets.unsqueeze(0).repeat(grid_coords.size(0), 1, 1)
        padded_coords = torch.clamp(padded_coords, min=0, max=grid_size - 1)

        # Convert padded coordinates back to indices
        padded_indices = padded_coords[:, :, 0] * grid_size + padded_coords[:, :, 1]

        return padded_indices

class ExpanderModule(nn.Module):
    def __init__(self, embed_dim=4096, hidden_dim=8192, num_heads=16, dropout=0.1, use_text=True, device='cuda:0'):
        super().__init__()
        self.use_text = use_text
        
        # Self-attention for selected patches
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        # Cross-attention between selected patches and remained patches
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm4 = nn.LayerNorm(embed_dim)
        
        # Final projection to get scores
        self.score_proj = nn.Linear(embed_dim, 1)

    def forward(self, selected_patches, remained_patches, text_embed=None):
        # Self-attention on selected patches
        #if self.use_text and text_embed is not None:
            # Cross-attention with remained patches and text
        concat_patches = torch.cat([selected_patches, text_embed], dim=0)
        x = self.self_attn(concat_patches, concat_patches, concat_patches)[0]
        x = self.norm1(x + concat_patches)

        y = self.self_attn_2(remained_patches, remained_patches, remained_patches)[0]
        remained_patches = self.norm3(y + remained_patches)
        
        x = self.cross_attn(remained_patches, x, x)[0]
        x = self.norm2(x + remained_patches)
        
        # FFN
        x_ffn = self.ffn(x)
        x = self.norm4(x + x_ffn)
        
        # Get scores for remained patches
        scores = self.score_proj(x).squeeze(-1)  # [num_remained_patches]
        scores = torch.sigmoid(scores)
        return scores

class CosSimRouter_pad_merge_learn(nn.Module):
    def __init__(self, model=None, gamma=0.06, noise_std=0.00, temperature=0.05, top_k=0, padding_size=1, embed_dim=4096, hidden_dim=8192, num_heads=16, dropout=0.1):
        super().__init__()
        self.gamma = gamma
        self.noise_std = noise_std
        self.temperature = temperature
        self.top_k = top_k
        self.padding_size = padding_size
        self.softmax = nn.Softmax(dim=0)
        self.expander = ExpanderModule(embed_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, use_text=False)
        
        # Extract the weights from the specified layers if provided
        self.q_proj_weight = None
        self.k_proj_weight = None
        if model is not None:
            for name, module in model.named_modules():
                if name == 'layers.0.self_attn.q_proj':
                    self.q_proj_weight = module.weight
                elif name == 'layers.0.self_attn.k_proj':
                    self.k_proj_weight = module.weight
            assert self.q_proj_weight is not None and self.k_proj_weight is not None, "Specified layers not found in the model"
        self.to("cuda:0")
        self.half()
    
    def spatially_uniform_sample(self, vision_feature, grid_size=24, sample_size=6):
        """
        Uniformly samples 'sample_size' tokens from each of the 'sample_size x sample_size' regions of the 'grid_size x grid_size' grid.
        """
        assert grid_size % sample_size == 0, "Grid size must be divisible by sample size for uniform sampling."

        region_size = grid_size // sample_size
        sampled_indices = []

        for i in range(sample_size):
            for j in range(sample_size):
                region_start = (i * region_size * grid_size) + (j * region_size)
                region_indices = region_start + torch.arange(0, region_size * grid_size, grid_size)[:, None] + torch.arange(0, region_size)
                region_indices = region_indices.flatten()
                sampled_index = region_indices[torch.randint(0, len(region_indices), (1,))].item()
                sampled_indices.append(sampled_index)

        sampled_indices = torch.tensor(sampled_indices, device=vision_feature.device)
        sampled_features = vision_feature[sampled_indices]
        return sampled_indices, sampled_features
    
    def forward(self, vision_feature, text_embed, attention_mask):
        L_v, D = vision_feature.shape
        L_t, D = text_embed.shape
        if self.q_proj_weight is not None:
            # Apply the extracted weights to the vision feature and text embedding
            vision_query = F.linear(vision_feature, self.q_proj_weight)  # (L_v, D)
            text_key = F.linear(text_embed, self.k_proj_weight)  # (L_t, D)
            # Calculate cosine similarity matrix
            cos_sim_matrix = F.cosine_similarity(vision_query.unsqueeze(1), text_key.unsqueeze(0), dim=-1)  # (L_v, L_t)
        else:
            cos_sim_matrix = F.cosine_similarity(vision_feature.unsqueeze(1), text_embed.unsqueeze(0), dim=-1)  # (L_v, L_t)
        
        # Apply attention mask to cosine similarity matrix
        cos_sim_matrix = cos_sim_matrix.masked_fill(attention_mask == False, 0.0)
        
        # Get the maximum value along the text (L_t) axis for each vision token
        scores, _ = cos_sim_matrix.max(dim=-1)  # (L_v,)
        
        # Add Gaussian noise to scores for diversity
        noise = torch.randn_like(scores) * self.noise_std
        scores = scores + noise
        
        # Apply temperature-scaled softmax to scores
        scores = self.softmax(scores/self.temperature)
        
        # Sort scores and get indices
        sorted_scores, indices = torch.sort(scores, descending=True)
        
        # Calculate cumulative sum of sorted scores
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        
        # Find the threshold index
        threshold_index = (cum_scores <= self.gamma).sum()
       
        # Select vision tokens up to the threshold index
        selected_indices = indices[:threshold_index]
        selected_vision_feature = vision_feature[selected_indices]
        
        if self.padding_size > 0:
            # Get the neighboring indices of the selected tokens
            selected_indices_grid = self.get_neighbor_indices(selected_indices, int(L_v ** 0.5), self.padding_size)
            
            # Flatten the selected indices and neighbor indices
            selected_indices_flat = selected_indices_grid.flatten()
            
            # Remove duplicate indices
            unique_indices = torch.unique(selected_indices_flat)
            
            # Gather the selected and neighboring vision tokens
            selected_vision_feature = vision_feature[unique_indices]
            selected_indices = unique_indices  # Update selected_indices to include neighbors
        
        # Score calculation for merging
        sampled_indices, sampled_features = self.spatially_uniform_sample(vision_feature)
        combined_indices = torch.cat([unique_indices, sampled_indices]).unique()
        combined_features = vision_feature[combined_indices]
        remaining_indices = torch.arange(L_v, device=vision_feature.device)[~torch.isin(torch.arange(L_v, device=vision_feature.device), combined_indices)]
        remaining_vision_feature = vision_feature[remaining_indices]
     
        
        if remaining_vision_feature.size(0) > 0:
            # Use the expander to calculate scores for merging
            merge_scores = self.expander(combined_features, remaining_vision_feature, text_embed)
            
            # Calculate cosine similarity between remaining and selected (with neighbors) patches
            cos_sim_remain_selected = F.cosine_similarity(
                remaining_vision_feature.unsqueeze(1), 
                combined_features.unsqueeze(0), 
                dim=-1
            )  # (L_r, L_s)
            
            # Find the selected patch with the highest cosine similarity for each remaining patch
            best_match_indices = torch.argmax(cos_sim_remain_selected, dim=1)  # (L_r,)
            
            # One-hot encoding of the best match indices
            one_hot_indices = F.one_hot(best_match_indices, num_classes=combined_features.size(0)).type_as(cos_sim_remain_selected)  # (L_r, L_s)
            
            # Apply one-hot indices to merge_scores to get selected_merge_scores for all selected patches
            selected_merge_scores = one_hot_indices * merge_scores.unsqueeze(-1)  # (L_r, L_s)
            selected_merge_scores = selected_merge_scores.type_as(remaining_vision_feature)
            
            # Add a row of ones to represent the self_score
            self_score = torch.ones(1, selected_merge_scores.size(1), device=selected_merge_scores.device)  # (1, L_s)
            selected_merge_scores = torch.cat([self_score, selected_merge_scores], dim=0)  # (1 + L_r, L_s)
            
            
            # Apply softmax to the non-zero elements along the first dimension (batch dimension)
            merge_weights = torch.zeros_like(selected_merge_scores)
            non_zero_mask = selected_merge_scores != 0
            
            # Flatten selected_merge_scores and apply softmax
            flat_scores = selected_merge_scores.flatten(start_dim=1)
            flat_mask = non_zero_mask.flatten(start_dim=1)
            
            # Avoid zero values by assigning large negative values before applying softmax
            flat_scores[~flat_mask] = -float('inf')
            merge_weights = F.softmax(flat_scores, dim=0).reshape_as(selected_merge_scores).to(remaining_vision_feature.dtype)
            
            # Extract self_weights (first row of merge_weights) and remain_weights (remaining rows)
            self_weights = merge_weights[0]
            
            remain_weights = merge_weights[1:]
            
            # Compute weighted sum of the merging tokens for all selected patches
            weighted_sum = torch.mm(remain_weights.T, remaining_vision_feature)  # (L_s, D)
            
            # Compute the self_token
            self_token = self_weights.unsqueeze(-1) * combined_features  # (L_s, D)
            
            
            # Combine original selected patches and weighted sum
            new_selected_vision_feature = weighted_sum + self_token
        
        if self.top_k == 0:
            return new_selected_vision_feature
        
        
        else:
            # Calculate cosine similarity between selected vision tokens and all vision tokens
            selected_cos_sim_matrix = F.cosine_similarity(new_selected_vision_feature.unsqueeze(1), vision_feature.unsqueeze(0), dim=-1)  # (threshold_index, L_v)
            
            # Get top-k similar vision tokens for each selected token
            _, top_k_indices = selected_cos_sim_matrix.topk(self.top_k, dim=-1)  # (threshold_index, top_k)
            
            # Gather top-k similar vision tokens
            top_k_vision_feature = vision_feature[top_k_indices]  # (threshold_index, top_k, D)
            
            # Calculate softmax weights for top-k similar tokens
            top_k_cos_sim = selected_cos_sim_matrix.gather(1, top_k_indices)  # (threshold_index, top_k)
            top_k_weights = F.softmax(top_k_cos_sim, dim=-1)  # (threshold_index, top_k)
            
            # Compute weighted sum of top-k similar tokens
            weighted_sum = torch.sum(top_k_vision_feature * top_k_weights.unsqueeze(-1), dim=1)  # (threshold_index, D)
            
            return weighted_sum
    
    def get_neighbor_indices(self, indices, grid_size, padding_size):
        # Convert indices to grid coordinates
        grid_coords = torch.stack((indices // grid_size, indices % grid_size), dim=1)

        # Generate offsets for neighboring tokens based on padding size
        offsets = []
        for i in range(-padding_size, padding_size + 1):
            for j in range(-padding_size, padding_size + 1):
                if i == 0 and j == 0:
                    continue
                offsets.append([i, j])
        offsets = torch.tensor(offsets).to(indices.device)

        # Apply padding to the grid coordinates
        padded_coords = grid_coords.unsqueeze(1) + offsets.unsqueeze(0).repeat(grid_coords.size(0), 1, 1)
        padded_coords = torch.clamp(padded_coords, min=0, max=grid_size - 1)

        # Convert padded coordinates back to indices
        padded_indices = padded_coords[:, :, 0] * grid_size + padded_coords[:, :, 1]

        return padded_indices


    
class ExpanderModule_mask(nn.Module):
    def __init__(self, embed_dim=4096, hidden_dim=8192, num_heads=16, dropout=0.1, use_text=True, device='cuda:0'):
        super().__init__()
        self.use_text = use_text
        
        # Self-attention for concatenated selected patch and text embeddings
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-attention between remained patches and the output of self-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm4 = nn.LayerNorm(embed_dim)
        
        # Final projection to get scores
        self.score_proj = nn.Linear(embed_dim, 1)

    def forward(self, selected_patches, remained_patches, text_embed=None, attention_mask=None):
        L_s, L_r, D = remained_patches.shape
        L_t = text_embed.shape[1] if text_embed is not None else 0
        
        # Concatenate selected patches with text embeddings
        if self.use_text and text_embed is not None:
            concat_input = torch.cat([selected_patches, text_embed], dim=1)  # (L_s, 1+L_t, D)
        else:
            concat_input = selected_patches  # (L_s, 1, D)

        # Self-attention on concatenated output
        
        x = self.self_attn(concat_input, concat_input, concat_input)[0]  # (L_s, 1+L_t, D)
        x = self.norm1(x + concat_input)
        
        # Apply attention mask in the cross-attention layer
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.t().unsqueeze(2).expand(-1, -1, 1+L_t)  # (L_s, L_r, 1+L_t)
            expanded_attention_mask = expanded_attention_mask.repeat_interleave(self.self_attn.num_heads, dim=0)  # (num_heads*L_s, L_r, 1+L_t)
        
        # Cross-attention with remained patches as query
        x = self.cross_attn(remained_patches, x, x, attn_mask=expanded_attention_mask)[0]  # (L_s, L_r, D)
        x = self.norm2(x + remained_patches)
        
        # FFN and score projection
        x_ffn = self.ffn(x)
        x = self.norm4(x + x_ffn)
        scores = self.score_proj(x).squeeze(-1)  # (L_s, L_r)

        # Apply sigmoid to the scores
        scores = torch.sigmoid(scores)  # (L_s, L_r)
        
        # Add a score of 1 for the selected patch itself
        self_score = torch.ones(L_s, 1, device=scores.device)  # (L_s, 1)
        scores = torch.cat([self_score, scores], dim=1)  # (L_s, 1+L_r)

        # Apply mask to the scores
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(L_s, 1, device=attention_mask.device), attention_mask.t()], dim=-1)  # (L_s, 1+L_r)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))  # Apply mask before softmax

        return scores

class CosSimRouter_pad_merge_learn_local(nn.Module):
    def __init__(self, model=None, gamma=0.05, noise_std=0.00, temperature=0.05, top_k=0, padding_size=1, embed_dim=4096, hidden_dim=8192, num_heads=16, dropout=0.1):
        super().__init__()
        self.gamma = gamma
        self.noise_std = noise_std
        self.temperature = temperature
        self.top_k = top_k
        self.padding_size = padding_size
        self.expander = ExpanderModule_mask(embed_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, use_text=True)
        
        # Extract the weights from the specified layers if provided
        self.q_proj_weight = None
        self.k_proj_weight = None
        if model is not None:
            for name, module in model.named_modules():
                if name == 'layers.0.self_attn.q_proj':
                    self.q_proj_weight = module.weight
                elif name == 'layers.0.self_attn.k_proj':
                    self.k_proj_weight = module.weight
            assert self.q_proj_weight is not None and self.k_proj_weight is not None, "Specified layers not found in the model"
    
    def forward(self, vision_feature, text_embed, attention_mask):
        L_v, D = vision_feature.shape
        L_t, D = text_embed.shape
        if self.q_proj_weight is not None:
            # Apply the extracted weights to the vision feature and text embedding
            vision_query = F.linear(vision_feature, self.q_proj_weight)  # (L_v, D)
            text_key = F.linear(text_embed, self.k_proj_weight)  # (L_t, D)
            # Calculate cosine similarity matrix
            cos_sim_matrix = F.cosine_similarity(vision_query.unsqueeze(1), text_key.unsqueeze(0), dim=-1)  # (L_v, L_t)
        else:
            cos_sim_matrix = F.cosine_similarity(vision_feature.unsqueeze(1), text_embed.unsqueeze(0), dim=-1)  # (L_v, L_t)
        
        # Apply attention mask to cosine similarity matrix
        cos_sim_matrix = cos_sim_matrix.masked_fill(attention_mask == False, 0.0)
        
        # Get the maximum value along the text (L_t) axis for each vision token
        scores, _ = cos_sim_matrix.max(dim=-1)  # (L_v,)
        
        # Add Gaussian noise to scores for diversity
        noise = torch.randn_like(scores) * self.noise_std
        scores = scores + noise
        
        # Apply temperature-scaled softmax to scores
        scores = F.softmax(scores / self.temperature, dim=0)
        
        # Sort scores and get indices
        sorted_scores, indices = torch.sort(scores, descending=True)
        
        # Calculate cumulative sum of sorted scores
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        
        # Find the threshold index
        threshold_index = (cum_scores <= self.gamma).sum()
       
        # Select vision tokens up to the threshold index
        selected_indices = indices[:threshold_index]
        selected_vision_feature = vision_feature[selected_indices].unsqueeze(1)  # (L_s, 1, D)
        
        if self.padding_size > 0:
            # Get the neighboring indices of the selected tokens
            selected_indices_grid = self.get_neighbor_indices(selected_indices, int(L_v ** 0.5), self.padding_size)
            
            # Flatten the selected indices and neighbor indices
            selected_indices_flat = selected_indices_grid.flatten()
            
            # Remove duplicate indices
            unique_indices = torch.unique(selected_indices_flat)
            
            # Gather the selected and neighboring vision tokens
            selected_vision_feature = vision_feature[unique_indices]
            expanded_selected_vision_feature = vision_feature[unique_indices].unsqueeze(1)  # (L_s, 1, D)
            selected_indices = unique_indices  # Update selected_indices to include neighbors
        
        # Score calculation for merging
        remaining_indices = torch.arange(L_v, device=vision_feature.device)[~torch.isin(torch.arange(L_v, device=vision_feature.device), selected_indices)]
        remaining_vision_feature = vision_feature[remaining_indices]
        expanded_remaining_vision_feature = vision_feature[remaining_indices].unsqueeze(0).expand(selected_vision_feature.size(0), -1, -1) # (L_s, L_r, D)
        if remaining_vision_feature.size(0) > 0:
            # Calculate cosine similarity between remaining and selected (with neighbors) patches
            cos_sim_remain_selected = F.cosine_similarity(remaining_vision_feature.unsqueeze(1), selected_vision_feature.unsqueeze(0), dim=-1)  # (L_s, L_r)
            
            # Find the selected patch with the highest cosine similarity for each remaining patch
            best_match_indices = torch.argmax(cos_sim_remain_selected, dim=1)  # (L_r,)
            
            # One-hot encoding of the best match indices as attention mask
            attention_mask = F.one_hot(best_match_indices, num_classes=selected_vision_feature.size(0)).type_as(cos_sim_remain_selected)  # (L_r, L_s)
            
            # Calculate the merge scores using the expander with the attention mask
            merge_scores = self.expander(expanded_selected_vision_feature, expanded_remaining_vision_feature, attention_mask=attention_mask)  # (L_s, 1+L_r)
            
            # Concatenate the selected patch with remaining patches
            all_patches = torch.cat([expanded_selected_vision_feature, expanded_remaining_vision_feature], dim=1)  # (L_s, 1+L_r, D)
            
            # Apply softmax to the merge scores
            merge_weights = F.softmax(merge_scores, dim=-1)  # (L_s, 1+L_r)
            merge_weights = merge_weights.type_as(all_patches)
            # Compute weighted sum of the merging tokens, including the selected patch itself
            weighted_sum = torch.bmm(merge_weights.unsqueeze(1), all_patches).squeeze(1)  # (L_s, D)
            
            # Update selected patches with weighted sum
            new_selected_vision_feature = weighted_sum  # (L_s, D)
            
        if self.top_k == 0:
            return new_selected_vision_feature
        
        else:
            # Calculate cosine similarity between selected vision tokens and all vision tokens
            selected_cos_sim_matrix = F.cosine_similarity(new_selected_vision_feature.unsqueeze(1), vision_feature.unsqueeze(0), dim=-1)  # (threshold_index, L_v)
            
            # Get top-k similar vision tokens for each selected token
            _, top_k_indices = selected_cos_sim_matrix.topk(self.top_k, dim=-1)  # (threshold_index, top_k)
            
            # Gather top-k similar vision tokens
            top_k_vision_feature = vision_feature[top_k_indices]  # (threshold_index, top_k, D)
            
            # Calculate softmax weights for top-k similar tokens
            top_k_cos_sim = selected_cos_sim_matrix.gather(1, top_k_indices)  # (threshold_index, top_k)
            top_k_weights = F.softmax(top_k_cos_sim, dim=-1)  # (threshold_index, top_k)
            
            # Compute weighted sum of top-k similar tokens
            weighted_sum = torch.sum(top_k_vision_feature * top_k_weights.unsqueeze(-1), dim=1)  # (threshold_index, D)
            
            return weighted_sum
    
    def get_neighbor_indices(self, indices, grid_size, padding_size):
        # Convert indices to grid coordinates
        grid_coords = torch.stack((indices // grid_size, indices % grid_size), dim=1)

        # Generate offsets for neighboring tokens based on padding size
        offsets = []
        for i in range(-padding_size, padding_size + 1):
            for j in range(-padding_size, padding_size + 1):
                if i == 0 and j == 0:
                    continue
                offsets.append([i, j])
        offsets = torch.tensor(offsets).to(indices.device)

        # Apply padding to the grid coordinates
        padded_coords = grid_coords.unsqueeze(1) + offsets.unsqueeze(0).repeat(grid_coords.size(0), 1, 1)
        padded_coords = torch.clamp(padded_coords, min=0, max=grid_size - 1)

        # Convert padded coordinates back to indices
        padded_indices = padded_coords[:, :, 0] * grid_size + padded_coords[:, :, 1]

        return padded_indices
                 
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.moe = True
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                self.moe
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes,
                self.router,
                self.moe
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
