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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

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
        print("Second Linear Layer Weight:", second_linear_weight)
        print("Second Linear Layer Bias:", second_linear_bias)
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

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def add_router(self):
        self.router = CosSimRouter_pad_merge_entropy()
    
    def get_router(self):
        router = getattr(self, 'router', None)
        return router
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def add_router(self):
        return self.get_model().add_router()
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def update_attention_mask(self, attention_mask, input_ids, labels):
        # Create new masks based on the conditions
        input_ids_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Find the index of the first occurrence of -200 in each sequence
        first_neg200_idx = (input_ids == -200).int().argmax(dim=1)
        
        # Mask the part on the left side of the first -200 in each sequence
        for i in range(input_ids.size(0)):
            if first_neg200_idx[i] > 0:
                input_ids_mask[i, :first_neg200_idx[i]+1] = False
        
        labels_mask = labels == -100
        
        # Combine the new masks with the original attention mask
        if labels==None:
            if attention_mask!=None:
                new_attention_mask = attention_mask & input_ids_mask
            else:
                new_attention_mask = input_ids_mask
            
        else:

            new_attention_mask = attention_mask & input_ids_mask & labels_mask
        
        return new_attention_mask


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, moe=False
    ):  
        router_attention_mask = self.update_attention_mask(attention_mask, input_ids, labels)
        
        new_input_ids = torch.where(input_ids == -200, torch.ones_like(input_ids), input_ids)
        
        text_embed = self.get_model().embed_tokens(new_input_ids)
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            
            split_sizes = [image.shape[0] for image in images]
            
            image_features = torch.split(image_features, split_sizes, dim=0)
            
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        if moe:
                            image_feature = self.get_model().get_router()(image_feature, text_embed, router_attention_mask)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                        if moe:
                            image_feature = self.get_model().get_router()(image_feature, text_embed, router_attention_mask)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
            
            if moe:
                if len(image_features.shape)==3:
                    
                    image_feature_list = []
                    if len(image_features)==1:
                        for i in range(len(image_features)):
                            cur_image_features = self.get_model().get_router()(image_features[i], text_embed[i], router_attention_mask[i])
                            image_feature_list.append(cur_image_features)
                    else:
                       
                        
                        for i in range(len(image_features)):
                            cur_image_features = self.get_model().get_router()(image_features[i], text_embed[0], router_attention_mask[0])
                            image_feature_list.append(cur_image_features)
                    image_features = image_feature_list
                    
                else:
                    image_features = self.get_model().get_router()(image_features, text_embed, router_attention_mask)
                
            

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        new_input_embeds = []
        #new_attention_mask = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            #cur_new_attention_mask = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                #cur_new_attention_mask.append(torch.ones((cur_input_embeds_no_im[i].shape[0],), device=attention_mask.device, dtype=attention_mask.dtype))
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    #cur_new_attention_mask.append(image_feature_masks[cur_image_idx])
                    
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            #cur_new_attention_mask = torch.cat(cur_new_attention_mask)
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            #new_attention_mask.append(new_attention_mask)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            #new_attention_mask = [x[:tokenizer_model_max_length] for x in new_attention_mask]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
