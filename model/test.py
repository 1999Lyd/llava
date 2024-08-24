import torch

def update_attention_mask(attention_mask, input_ids, labels):
    # Create new masks based on the conditions
    input_ids_mask = input_ids != -200
    labels_mask = labels == -100
    
    # Combine the new masks with the original attention mask
    new_attention_mask = attention_mask & input_ids_mask & labels_mask
    
    return new_attention_mask

# Example usage
B, L = 2, 5  # Batch size and sequence length
attention_mask = torch.tensor([[True, True, True, False, False],
                               [True, True, True, True, False]])
input_ids = torch.tensor([[1, 2, -200, 4, 5],
                          [1, 2, 3, -200, 5]])
labels = torch.tensor([[1, -100, 3, 4, 5],
                       [1, 2, -100, 4, 5]])

new_attention_mask = update_attention_mask(attention_mask, input_ids, labels)
print(new_attention_mask)