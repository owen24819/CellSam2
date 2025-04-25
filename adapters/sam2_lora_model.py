from pathlib import Path
import torch

from peft.tuners.lora import Linear as LoRALinear

from adapters.adapter_modules import ImageEncoderLoRAAdapter, RoPELoRAAdapter

class SAM2LoRAModel:
    def __init__(
        self,
        model,
        lora_rank=32,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_bias='none',
        adapt_encoder=True,
        adapt_memory=True,
        adapt_decoder=True,
        adapt_mlp=True,
        trainable_iou_pred_heads=True,
        trainable_obj_score_heads=True,
    ):
        self.model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.adapt_encoder = adapt_encoder
        self.adapt_memory = adapt_memory
        self.adapt_decoder = adapt_decoder
        self.adapt_mlp = adapt_mlp
        self.trainable_iou_pred_heads = trainable_iou_pred_heads
        self.trainable_obj_score_heads = trainable_obj_score_heads

        self.inject()

    def create_adapter(self, blk, dim):
        """Create a LoRA adapter for a given block.
        Used for image encoder becuase you can't access q,k,v directly."""

        return ImageEncoderLoRAAdapter(
            blk,
            dim,
            self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            bias=self.lora_bias
        )

    def inject_lora_to_rope_attention(self, attn_module, adapter_name):
        """Wrap a RoPEAttention block with RoPELoRAAdapter."""

        return RoPELoRAAdapter(
            attn_module,
            adapter_name=adapter_name,
            lora_rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            apply_lora_to_v=False,
            bias=self.lora_bias
        )
    
    
    def inject(self):

        """Inject LoRA adapters into the model."""

        if self.adapt_encoder:
            # Image encoder trunk
            for i, blk in enumerate(self.model.image_encoder.trunk.blocks):
                dim = blk.attn.qkv.in_features
                self.model.image_encoder.trunk.blocks[i].attn = self.create_adapter(blk.attn, dim)

        if self.adapt_memory:
            # Memory layers
            for i, layer in enumerate(self.model.memory_attention.layers):
                # Self-attention
                adapter_name = f'lora/memory_attention/layers-{i}/self_attn'
                layer.self_attn = self.inject_lora_to_rope_attention(layer.self_attn, adapter_name)

                # Cross-attention
                adapter_name = f'lora/memory_attention/layers-{i}/cross_attn_image'
                layer.cross_attn_image = self.inject_lora_to_rope_attention(layer.cross_attn_image, adapter_name)

                if self.adapt_mlp:
                    # Add LoRA to feed-forward networks
                    adapter_name = f'lora/memory_attention/layers-{i}/linear2'
                    layer.linear2 = LoRALinear(
                        base_layer=layer.linear2,
                        adapter_name=adapter_name,
                        r=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout
                        )

        if self.adapt_decoder:
            for i, layer in enumerate(self.model.sam_mask_decoder.transformer.layers):
                # Attention
                adapter_name = f'lora/sam_mask_decoder/transformer/layers-{i}/self_attn'
                layer.self_attn = self.inject_lora_to_rope_attention(layer.self_attn, adapter_name)

                # Cross-attention - token to image
                adapter_name = f'lora/sam_mask_decoder/transformer/layers-{i}/cross_attn_token_to_image'
                layer.cross_attn_token_to_image = self.inject_lora_to_rope_attention(layer.cross_attn_token_to_image, adapter_name)

                # Cross-attention - image to token
                adapter_name = f'lora/sam_mask_decoder/transformer/layers-{i}/cross_attn_image_to_token'
                layer.cross_attn_image_to_token = self.inject_lora_to_rope_attention(layer.cross_attn_image_to_token, adapter_name)
                
                if self.adapt_mlp:
                    # Add LoRA to feed-forward networks
                    adapter_name = f'lora/sam_mask_decoder/transformer/layers-{i}/mlp/layers-1'
                    layer.mlp.layers[1] = LoRALinear(
                        base_layer=layer.mlp.layers[1],
                        adapter_name=adapter_name,
                        r=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout
                    )

            # Cross-attention - image to token
            adapter_name = 'lora/sam_mask_decoder/transformer/final_attn_token_to_image'
            self.model.sam_mask_decoder.transformer.final_attn_token_to_image = self.inject_lora_to_rope_attention(self.model.sam_mask_decoder.transformer.final_attn_token_to_image, adapter_name)

        # Freeze all parameters by default
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters and optionally prediction heads
        trainable_patterns = ['lora']
        if self.trainable_iou_pred_heads:
            trainable_patterns.append('iou_prediction_head')
        if self.trainable_obj_score_heads:
            trainable_patterns.append('pred_obj_score_head')
            
        for name, param in self.model.named_parameters():
            if any(pattern in name.lower() for pattern in trainable_patterns):
                param.requires_grad = True

        self.print_trainable()

    def save_adapters(self, save_dir):

        """Save the LoRA adapters to a file."""

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        lora_state_dict = {
            k: v
            for k, v in self.model.state_dict().items()
            if "lora/" in k or "prompt" in k
        }
        save_path = save_dir / "lora_adapters.pth"
        torch.save(lora_state_dict, save_path)
        print(f"✅ Saved LoRA adapters to: {save_path}")

    def print_trainable(self):

        """Print the trainable LoRA parameters."""

        print("\n🔍 Trainable LoRA parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Trainable:  {name} | shape: {tuple(param.shape)}")

    def load_adapters(self, adapter_path):

        """Load the LoRA adapters from a file."""

        # Sanity check
        lora_keys_in_model = any("lora/" in k or "prompt" in k for k, _ in self.model.named_parameters())
        if not lora_keys_in_model:
            print("⚠️ Warning: No LoRA modules found in the model before loading. Did you forget to call `inject()`?")

        adapter_path = Path(adapter_path)
        assert adapter_path.exists(), f"Adapter file not found: {adapter_path}"
        lora_state_dict = torch.load(adapter_path, map_location="cpu")

        # Only load matching keys (optional but safer during experimentation)
        missing_keys, unexpected_keys = self.model.load_state_dict(lora_state_dict, strict=False)

        print(f"✅ Loaded LoRA adapters from {adapter_path}")
        if missing_keys:
            print(f"⚠️ Missing keys (not loaded): {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys (ignored): {unexpected_keys}")
