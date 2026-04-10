import numpy as np


def clip_very_high_values(attn, threshold):
   
    vmax = np.percentile(attn, threshold)  
    attn_clipped = np.clip(attn, a_min=None, a_max=vmax)
    vmin = attn_clipped.min()
    attn_normalized = (attn_clipped - vmin) / (attn_clipped.max() - vmin)
    return attn_normalized

def average_division_x_entropy(x, attn_np):
    attention_array = attn_np.flatten()
    if x <= 0:
        raise ValueError("分类数x必须为正整数")
    if not isinstance(attention_array, np.ndarray) or len(attention_array.shape) != 1:
        raise ValueError("attention_array必须是一维numpy数组")
    if len(attention_array) == 0:
        raise ValueError("attention数组不能为空")

    min_val = np.min(attention_array)
    max_val = np.max(attention_array)
    edges = np.linspace(min_val, max_val, x + 1)

    counts, _ = np.histogram(attention_array, bins=edges)
    total = len(attention_array)
    probabilities = counts / total
    non_zero_probs = probabilities[probabilities > 0]
    entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))

    return entropy



