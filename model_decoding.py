"""
Implementation of the Dewave model for EEG-to-Text decoding.
This module contains the main model architecture and its components.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import math
import numpy as np
from types_ import *
from VQVAE import VectorQuantizer
from conformer import ConformerBlock
from types_ import *
from util import checkpoint


# L_{\text{contrast}} = -\frac{1}{n} \sum \log [\frac{\exp(s_{ii} / \tau)}{\sum_{k=1}^{N} \exp(s_{ik} / \tau)} ], \quad s_{ij} = \mathbf{z}_q(\mathbf{x}^i)^T \mathbf{z}_t(j)
def cos_sim(A, B, dim: int = -1, eps: float = 1e-4):
    # ℓ2‑norm
    norm_A = F.normalize(A, p=2, dim=dim, eps=eps)
    norm_B = F.normalize(B, p=2, dim=dim, eps=eps)
    # No longer clamp(min=0) - negative similarities are preserved
    return torch.matmul(norm_A, norm_B.transpose(-1, -2))


class ContrastiveLatentLoss(nn.Module):
    def __init__(self, gamma: float = 0.07):
        """
        Contrastive loss for aligning EEG Condition with Text Latent in Latent Diffusion Model (LDM).
        """
        super(ContrastiveLatentLoss, self).__init__()
        self.gamma = gamma
        # 额外内部常量：安全剪裁阈值，不改变外部接口
        self._clip_val = 50.0

    def forward(self, text_latent: torch.Tensor,
                eeg_condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_latent (torch.Tensor): (batch_size, seq_len, hidden_dim)
            eeg_condition (torch.Tensor): (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = text_latent.shape

        # Normalization (keep the original variable name)
        E = F.normalize(eeg_condition, dim=-1)
        T = F.normalize(text_latent, dim=-1)

        # Calculate similarity and temperature scaling
        similarity = torch.matmul(E, T.transpose(1, 2)) / self.gamma   # (B,S,S)

        # Re: clip to [-_clip_val, +_clip_val] to prevent exp overflow
        similarity = similarity.clamp(min=-self._clip_val,
                                      max=+self._clip_val)

        # Generate diagonal labels
        labels = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1) \
                 .to(eeg_condition.device)          # shape (B,S)

        # Change to 2-D cross entropy calculation
        similarity_flat = similarity.reshape(batch_size * seq_len, seq_len)
        labels_flat = labels.reshape(batch_size * seq_len)

        loss = F.cross_entropy(similarity_flat, labels_flat)
        return loss
        # 归一化
        # E = F.normalize(eeg_condition, dim=-1)
        # T = F.normalize(text_latent, dim=-1)

        # # 1) Text→EEG similarity, shape [B, S, S]
        # sim_te = torch.matmul(T, E.transpose(1,2)) / self.gamma
        # # 2) EEG→Text similarity (transpose the inputs)
        # sim_et = torch.matmul(E, T.transpose(1,2)) / self.gamma

        # # 裁剪，防止 exp 溢出
        # sim_te = sim_te.clamp(-self._clip_val, self._clip_val)
        # sim_et = sim_et.clamp(-self._clip_val, self._clip_val)

        # # 构造对角标签：每个句子里第 i 个 token 应与第 i 个对齐
        # batch_size, seq_len, _ = text_latent.shape
        # labels = torch.arange(seq_len, device=sim_te.device).unsqueeze(0).repeat(batch_size,1)

        # # 展平
        # sim_te_flat = sim_te.reshape(batch_size*seq_len, seq_len)
        # sim_et_flat = sim_et.reshape(batch_size*seq_len, seq_len)
        # labels_flat = labels.reshape(batch_size*seq_len)

        # # 计算两个方向的交叉熵
        # loss_te = F.cross_entropy(sim_te_flat, labels_flat)
        # loss_et = F.cross_entropy(sim_et_flat, labels_flat)

        # # 返回对称 InfoNCE loss
        # return 0.5 * (loss_te + loss_et)       
    

    
class rawEEGEmbedder(nn.Module):
    def __init__(self, seq_len: int = 56, channel: int = 105, 
                 dim_feedforward: int = 512, d_model: int = 512, 
                 num_heads: int = 4, num_layers: int = 3) -> None:
        """
        Raw EEG signal embedder using CNN and Transformer.

        Args:
            seq_len: Length of input sequence
            channel: Number of EEG channels
            dim_feedforward: Dimension of feedforward network
            d_model: Dimension of model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super(rawEEGEmbedder, self).__init__()
        convs = []
        in_channels = channel   # 단일 채널 음성
        layer_defs = [
            # (out_ch, kernel_size, stride, padding)
            (64, 10, 5, 1),
            (64, 3, 2, 1),
            (128, 3, 2, 1),
            (128, 3, 2, 1),
            (512, 3, 2, 1),
            (512, 2, 2, 0),
        ]

        # Build 6-layer CNN
        # 6-layer CNN encoder slides through the whole wave and gets the embedding sequence.
        for (oc, k, s, p) in layer_defs:
            convs.append(nn.Conv1d(
                in_channels, oc, kernel_size=k, stride=s,
                padding=p, bias=True
            ))
            convs.append(nn.GELU())
            convs.append(nn.Dropout(p=0.1))
            in_channels = oc
        self.conv_layers = nn.Sequential(*convs)

        # Transformer encoder for sequence modeling
        # A transformer layer with head number 8
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,  # 임의
            batch_first=True  # [B, S, C] 모드
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # 1 × 1 convolutional layer are combined to fuse multiple EEG channels into one embedding with size 512.
        self.conv1x1 = nn.Conv1d(
            in_channels=seq_len,   # 입력 채널
            out_channels=seq_len, # 출력 채널
            kernel_size=1,            # 1x1 커널
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the raw EEG embedder.

        Args:
            x: Input tensor of shape (batch_size, channels, seq_len, time)

        Returns:
            torch.Tensor: Embedded representation
        """
        return checkpoint(
            self._forward, (x, ), self.parameters(), True
        )
    
    def _forward(self, x):
        B, C, S, T = x.shape
        x = x.reshape(B * S, C, T)

        conv_out = self.conv_layers(x)

        conv_out = conv_out.reshape(B, S, -1)

        transformer_out = self.transformer(conv_out)
        out = self.conv1x1(transformer_out)
        return out

"""
Theta band (5-7Hz), the Alpha band (8-13Hz), the Beta band (12-30Hz), and Gamma band (30Hz-) [27] to get the statistic frequency features of each fragment. 
It is noted that although different fragments may have different EEG window sizes, the statistical results are the same (embedding size 840).

56 tokens each with an 840 embedding size

transformer layer with head number 8 and a 1 × 1 convolutional layer are combined to fuse multiple EEG channels into an embedding sequence with size 512

"""
class EEGEmbedder(nn.Module):
    def __init__(self, in_feature: int = 840, decoder_embedding_size: int = 512, 
                 additional_encoder_nhead: int = 4, 
                 additional_encoder_dim_feedforward: int = 512) -> None:
        """
        EEG feature embedder using Transformer.

        Args:
            in_feature: Input feature dimension
            decoder_embedding_size: Size of decoder embeddings
            additional_encoder_nhead: Number of attention heads in additional encoder
            additional_encoder_dim_feedforward: Feedforward dimension in additional encoder
        """
        super(EEGEmbedder, self).__init__()
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, 
            nhead=additional_encoder_nhead,  
            dim_feedforward = additional_encoder_dim_feedforward, 
            batch_first=True)
        
        self.additional_encoder = nn.TransformerEncoder(
            self.additional_encoder_layer, 
            num_layers=6)

        self.fc1 = nn.Linear(in_feature, 1024)

    def forward(self, input_embeddings_batch: torch.Tensor, 
                input_masks_invert: torch.Tensor) -> torch.Tensor:
        return checkpoint(
            self._forward, (input_embeddings_batch,  input_masks_invert), self.parameters(), True
        )
    def forward(self, input_embeddings_batch: torch.Tensor, 
                input_masks_invert: torch.Tensor) -> torch.Tensor:
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""

        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch)
        # use src_key_padding_masks
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)

        # encoded_embedding = self.additional_encoder(input_embeddings_batch)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding


"""
For structure-wise, a conformer-based multi-layer encoder with specially designed hyperparameters is employed. 
The one-dimensional convolution layer processes the EEG waves to generate the embedding sequence 4, fusing the EEG channels into a unique embedding for each period. 
We apply bi-directional transformer attention layers to the sequence to capture temporal relations.
"""

# Conformer: kernel (10,3,3,3,2), stride(3,2,2,2,2)
# Codex Transformer: head=8, dim=512, layer=6
class Encoder(nn.Module):
    def __init__(self):
        """
        Encoder module using Conformer blocks and additional transformer encoder.
        """
        super(Encoder, self).__init__()
        kernel_sizes = [10, 3, 3, 3, 2]
        stride_sizes = [3,2,2,2,2]

        # self.conformer = ConformerBlock(
        #     encoder_dim=512,
        #     num_attention_heads=4,
        #     feed_forward_expansion_factor=4, # error :feed_forward_expansion_factor=512,
        #     conv_kernel_size = kernel_size,
        #     stride_size = stride_size
        # )
        # Multi-layer Conformer
        num_conformer_layers = len(kernel_sizes)  # 5 blocks
        self.conformers = nn.Sequential(*[
            ConformerBlock(
                encoder_dim=512,
                num_attention_heads=4,
                feed_forward_expansion_factor=4,
                conv_expansion_factor=2,
                feed_forward_dropout_p=0.1,
                attention_dropout_p=0.1,
                conv_dropout_p=0.1,
                conv_kernel_size=kernel_sizes[i],  # 单值
                stride_size=stride_sizes[i],       # 单值
                half_step_residual=True
            )
            for i in range(num_conformer_layers)
        ])

        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8,  dim_feedforward = 512, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Encoded representation
        """
        return checkpoint(
            self._forward, (x, ), self.parameters(), True
        )

    def _forward(self, x):
        conv_out = self.conformers(x)
        out = self.additional_encoder(conv_out)
        return out
    
# Codex Transformer: head=8, dim=512, layer=6
# CNNs: kernel (3,3,3), stride(2,2,3)
# TransposeCNN: kernel (3), stride(2)
class Decoder(nn.Module):
    def __init__(self):
        """
        Decoder module using transformer encoder and transposed convolutions.
        """
        super(Decoder, self).__init__()
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=8,  
            dim_feedforward = 512, 
            batch_first=True)

        self.additional_encoder = nn.TransformerEncoder(
            self.additional_encoder_layer, 
            num_layers=6)
        
        layers = []
        prev_filters = 512
        # Build a sequence of 6 convolution layers
        layer_defs = [
            # (out_ch, kernel_size, stride, padding)
            (256, 3, 2, 2),
            (128, 3, 2, 2),
            (64, 3, 3, 2),
        ]
        for (oc, k, s, p) in layer_defs:
            layers.append(
                nn.Conv1d(
                    in_channels=prev_filters,
                    out_channels=oc,
                    kernel_size=k,
                    padding=p,
                    stride=s
                ), 
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=0.1))
            prev_filters = oc

        self.conv_stack = nn.Sequential(*layers)

        self.up = nn.ConvTranspose1d(64, 512, kernel_size=3,stride=2)
        self.projector = nn.Linear(13, 56)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Decoded representation
        """
        return checkpoint(
            self._forward, (x, ), self.parameters(), True
        )

    def _forward(self, x):
        transformer_out = self.additional_encoder(x)
        transformer_out = transformer_out.permute(0,2,1).contiguous()
        conv_out = self.conv_stack(transformer_out)
        out = self.up(conv_out)
        out = self.projector(out)
        out = out.permute(0,2,1).contiguous()
        return out

class Dewave(nn.Module):
    def __init__(
        self,
        pretrained_layer: nn.Module,
        input_type: str,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dims: Optional[List] = None,
        beta: float = 0.25,
        in_feature: int = 840, 
        decoder_embedding_size: int = 512, 
        additional_encoder_nhead: int = 8, 
        additional_encoder_dim_feedforward: int = 2048,
        **kwargs
    ) -> None:
        """
        Dewave model for EEG-to-Text decoding.

        Args:
            pretrained_layer: Pre-trained language model
            input_type: Type of input ('rawEEG' or 'features')
            embedding_dim: Dimension of embeddings
            num_embeddings: Number of embeddings in VQ layer
            hidden_dims: Dimensions of hidden layers
            beta: Commitment loss weight
            in_feature: Input feature dimension
            decoder_embedding_size: Size of decoder embeddings
            additional_encoder_nhead: Number of attention heads
            additional_encoder_dim_feedforward: Feedforward dimension
        """
        super(Dewave, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        
        self.projector = nn.Linear(pretrained_layer.config.d_model, 512)

        # Initialize appropriate embedder based on input type
        if input_type == 'rawEEG':
            self.embedder = rawEEGEmbedder()
        else:
            self.embedder = EEGEmbedder()

        # Initialize encoder, decoder and other components
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.language_model = pretrained_layer
        
        # ———————————— frozen BART Encoder（Only fine-tune Decoder） ————————————
        for name, param in self.language_model.named_parameters():
            if "decoder" not in name:
                param.requires_grad = False

        # DeWave uses a codex with size 2048
        # codex latent is an embedding with size 512.
        self.vq_layer = VectorQuantizer(
            num_embeddings,
            embedding_dim,
            self.beta)
        
        self.contrastive_learning = ContrastiveLatentLoss()

        # Project language model dimension to model dimension
        self.fc = nn.Linear(512, 1024)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using the encoder.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Encoded representation
        """
        out = self.encoder(x)
        return out

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized representation.

        Args:
            z_q: Quantized representation

        Returns:
            torch.Tensor: Decoded output
        """
        out = self.decoder(z_q)

        return out


    def forward(self, x: torch.Tensor, text: torch.Tensor, 
                input_masks_batch: torch.Tensor, 
                target_ids_batch: torch.Tensor, 
                stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Dewave model.

        Args:
            x: Input EEG tensor
            text: Input text tensor
            input_masks_batch: Input masks
            target_ids_batch: Target IDs
            stage: Current stage ('train' or 'eval')

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model outputs and loss
        """
        x = self.embedder(x)

        z_c = self.encode(x)

        z_q, vq_loss = self.vq_layer(z_c)
        
        text_emb = self.language_model.model.encoder.embed_tokens(text)
        text_emb = self.projector(text_emb)

        if stage == 'pretrained':
            out = self.decode(z_q)
            recon_loss = F.mse_loss(out, x)
            loss = self.loss_function(text_embedding = text_emb,eeg_embedding = z_q, recon_loss = recon_loss, vq_loss=vq_loss)
        else:
            out = self.fc(z_q)
            out = self.language_model(inputs_embeds = out, attention_mask = input_masks_batch,
                                            return_dict = True, labels = target_ids_batch)
            loss = self.loss_function(recon_loss = out.loss, vq_loss = vq_loss)
            # loss = self.loss_function(recon_loss = out.loss, vq_loss = vq_loss, text_embedding= text_emb, eeg_embedding = z_q)
  
        return out, loss
        # return out
    
    @torch.no_grad()
    def generate(
        self,
        input_embeddings_batch: torch.Tensor,
        input_masks_batch: torch.Tensor,
        input_masks_invert: torch.Tensor,
        target_ids_batch_converted: torch.Tensor,
        dummy_decoder_inputs_ids: torch.Tensor,
        generation_config: Optional[Any] = None,
        logits_processor: Optional[Any] = None,
        stopping_criteria: Optional[Any] = None,
        prefix_allowed_tokens_fn: Optional[Callable] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional[Any] = None,
        streamer: Optional[Any] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text from EEG embeddings using the model.

        Args:
            input_embeddings_batch (torch.Tensor): Batch of input EEG embeddings
            input_masks_batch (torch.Tensor): Attention masks for the input
            input_masks_invert (torch.Tensor): Inverted attention masks
            target_ids_batch_converted (torch.Tensor): Converted target token IDs
            generation_config (Optional[Any]): Configuration for text generation
            logits_processor (Optional[Any]): Processor for output logits
            stopping_criteria (Optional[Any]): Criteria for stopping generation
            prefix_allowed_tokens_fn (Optional[Callable]): Function for allowed prefix tokens
            synced_gpus (Optional[bool]): Whether to sync across GPUs
            assistant_model (Optional[Any]): Assistant model for generation
            streamer (Optional[Any]): Streamer for generation output
            negative_prompt_ids (Optional[torch.Tensor]): IDs for negative prompts
            negative_prompt_attention_mask (Optional[torch.Tensor]): Attention masks for negative prompts
            device (str): Device to run generation on
            **kwargs: Additional arguments for generation

        Returns:
            torch.Tensor: Generated token I Ds
        """

        # Process through embedder
        x = self.embedder(input_embeddings_batch)

        # Encode the embeddings
        z_c = self.encode(x)

        # Apply vector quantization
        z_q, vq_loss = self.vq_layer(z_c)
        out = self.fc(z_q)
        
        output=self.language_model.generate(
            inputs_embeds = out,
            attention_mask = input_masks_batch[:,:out.shape[1]],
            decoder_input_ids = dummy_decoder_inputs_ids,
            # labels = target_ids_batch_converted,
            # generation_config=generation_config,
            # logits_processor=logits_processor,
            # stopping_criteria=stopping_criteria,
            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            # synced_gpus=synced_gpus,
            # assistant_model=assistant_model,
            # streamer=streamer,
            # negative_prompt_ids=negative_prompt_ids,
            # negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,)
        return output


    def loss_function(
        self,
        recon_loss: torch.Tensor,
        text_embedding: Optional[torch.Tensor] = None,
        eeg_embedding: Optional[torch.Tensor] = None,
        vq_loss: Optional[torch.Tensor] = None,
        alpha: float = 0.1
    ) -> torch.Tensor:
        """
        Calculate the total loss combining reconstruction, VQ, and contrastive losses.

        Args:
            recon_loss (torch.Tensor): Reconstruction loss
            text_embedding (Optional[torch.Tensor]): Text embeddings for contrastive loss
            eeg_embedding (Optional[torch.Tensor]): EEG embeddings for contrastive loss
            vq_loss (Optional[torch.Tensor]): Vector quantization loss
            alpha (float): Weight factor for contrastive loss

        Returns:
            torch.Tensor: Combined loss value
        """
        
        
        # L_{\text{wave}} = \frac{1}{n} \sum  (\phi(\mathbf{z}_q(\mathcal{X}) - \mathcal{X})^2_n + \| \text{sg}[\mathbf{z}_c(\mathbf{x})] - \mathbf{z}_q(\mathbf{x}) \|_2^2 + \beta \| \mathbf{z}_c(\mathbf{x}) - \text{sg}[\mathbf{z}_q(\mathbf{x})] \|_2^2 
        if text_embedding != None or eeg_embedding != None:
            loss = recon_loss + vq_loss + alpha * self.contrastive_learning(text_embedding, eeg_embedding)
        # Use only reconstruction and VQ losses
        else:
            loss = recon_loss + vq_loss

        return loss