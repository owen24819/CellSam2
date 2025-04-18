import torch.nn as nn
from peft.tuners.lora import Linear as LoRALinear

class ImageEncoderLoRAAdapter(nn.Module):
    def __init__(
        self, 
        blk, 
        dim, 
        lora_rank=32,
        alpha=8,  # Scaling factor for LoRA
        dropout=0.1,  # Dropout probability
        bias='none',  # Bias configuration
    ):
        super().__init__()
        self.block = blk
        self.scaling = alpha / lora_rank  # Scale factor as described in LoRA paper
        self.dropout = nn.Dropout(p=dropout)

        use_bias1 = bias in ('all', 'lora_only')
        use_bias2 = bias == 'all'

        self.lora_adapter = nn.Sequential(
            nn.Linear(dim, lora_rank, bias=use_bias1),
            nn.GELU(),
            nn.Linear(lora_rank, dim, bias=use_bias2)
        )

    def forward(self, x, *args, **kwargs):
        x = x + self.dropout(self.scaling * self.lora_adapter(x))
        return self.block(x, *args, **kwargs)
    

class RoPELoRAAdapter(nn.Module):
    def __init__(
        self,
        attn_module,
        adapter_name,
        lora_rank=32,
        alpha=8,  # Scaling factor for LoRA
        dropout=0.1,  # Dropout probability
        apply_lora_to_v=False,
        bias='none',
    ):
        super().__init__()
        self.attn_module = attn_module

        self.attn_module.q_proj = LoRALinear(
            base_layer=self.attn_module.q_proj,
            adapter_name=f"{adapter_name}_q_proj",
            r=lora_rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias
        )

        self.attn_module.k_proj = LoRALinear(
            base_layer=self.attn_module.k_proj,
            adapter_name=f"{adapter_name}_k_proj",
            r=lora_rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias
        )

        if apply_lora_to_v:
            self.attn_module.v_proj = LoRALinear(
                base_layer=self.attn_module.v_proj,
                adapter_name=f"{adapter_name}_v_proj",
                r=lora_rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias=bias
            )

    def forward(self, *args, **kwargs):
        return self.attn_module(*args, **kwargs)
