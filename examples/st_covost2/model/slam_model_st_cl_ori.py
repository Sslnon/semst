import os
import types
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import ot
import numpy as np

from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.train_utils import print_module_size, print_model_size
from peft import PeftModel, PeftConfig
from torch.nn import CrossEntropyLoss
from slam_llm.utils.metric import compute_accuracy
from geomloss import SamplesLoss
import logging
logger = logging.getLogger(__name__)

import torch.nn as nn
# sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05)
def span_pooling(x: torch.Tensor, w: int = 3, stride: int = 3, mode: str = "mean"):
    T, D = x.shape
    spans = []
    for i in range(0, T - w + 1, stride):
        chunk = x[i:i+w]
        if mode == "mean":
            pooled = chunk.mean(dim=0)
        elif mode == "max":
            pooled = chunk.max(dim=0).values
        spans.append(pooled)
    if len(spans) == 0:
        return x.mean(dim=0, keepdim=True)  # fallback if too short
    return torch.stack(spans, dim=0)
# projector.py
import torch, torch.nn as nn

class LSARProjector(nn.Module):
    def __init__(self, P_global: torch.Tensor, freeze=True):
        super().__init__()
        self.P = nn.Parameter(P_global, requires_grad=not freeze)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(h, self.P)          # [B,Q,k]
        return h - torch.matmul(proj, self.P.t())

sinkhorn = SamplesLoss(loss="sinkhorn",p=2,blur=0.05)
def sinkhorn_wasserstein_loss(x_seq, y_seq):
    return sinkhorn(x_seq, y_seq)
# def compute_ot_plan_pot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """
#     x, y: [Q, D] float32
#     Returns Z_ot: [Q, Q] torch.Tensor
#     """
#     x_np = x.detach().float().cpu().numpy()
#     y_np = y.detach().float().cpu().numpy()
#     Q = x_np.shape[0]
#     a = b = np.ones(Q) / Q
#     C = ot.dist(x_np, y_np, metric="euclidean") ** 2  # [Q, Q]
#     Z = ot.emd(a, b, C)
#     return torch.from_numpy(Z).to(x.device).float()
# def _sync_bn(num_feat: int, affine: bool = True):

#     return nn.SyncBatchNorm(num_feat, affine=affine)

# class AlignmentHead(nn.Module):
#     def __init__(self, dim: int):
#         super().__init__()
#         self.q_proj = nn.Linear(dim, dim, bias=False)
#         self.k_proj = nn.Linear(dim, dim, bias=False)
#         self.scale  = dim ** 1

#     def forward(self, src_tok: torch.Tensor, tgt_tok: torch.Tensor):  # [B, Q, D]
        
#         q = self.q_proj(src_tok)
#         k = self.k_proj(tgt_tok)

#         sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         z_hat = sim.softmax(dim=-1)

#         # print("sim range:", sim.min().item(), sim.max().item())
#         # print("sim std:", sim.std().item())
#         # print("z_hat entropy:", (-z_hat * z_hat.log()).sum(dim=-1).mean().item())  # 越大说明越“平”

#         return z_hat

# def gumbel_sinkhorn_sample(C: torch.Tensor, temperature: float = 0.1, n_iters: int = 20):
#     noise = -torch.empty_like(C).exponential_().log()  # Gumbel(0,1)
#     logits = -C + noise
#     Z = logits / temperature
#     for _ in range(n_iters):
#         Z = Z - torch.logsumexp(Z, dim=1, keepdim=True)
#         Z = Z - torch.logsumexp(Z, dim=0, keepdim=True)
#     return Z.exp()

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    encoder = setup_encoder(train_config, model_config, **kwargs)

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )
    model = slam_model(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get("ckpt_path", None) #FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
            logger.info("loading other parts from: {}".format(ckpt_path))
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            module_dict=ckpt_dict['module']
            model.load_state_dict(module_dict, strict=False)
    print_model_size(model, train_config, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model, tokenizer


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    if "vallex" in model_config.llm_name.lower():
        return None  
    elif "mupt" in model_config.llm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path,
                                            trust_remote_code=True,
                                            use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    encoder_list = model_config.encoder_name.split(",") if model_config.encoder_name else []
    if len(encoder_list) == 0:
        return None
    if len(encoder_list) == 1:
        encoder_name = encoder_list[0]
        if encoder_name == "whisper" or encoder_name == "qwen-audio":
            from slam_llm.models.encoder import WhisperWrappedEncoder
            encoder = WhisperWrappedEncoder.load(model_config)
        if encoder_name == "beats": 
            from slam_llm.models.encoder import BEATsEncoder
            encoder = BEATsEncoder.load(model_config)
        if encoder_name == "eat":
            from slam_llm.models.encoder import EATEncoder
            encoder = EATEncoder.load(model_config)
        if encoder_name == "SpatialAST":
            from slam_llm.models.encoder import SpatialASTEncoder
            encoder = SpatialASTEncoder.load(model_config)
        if encoder_name == "wavlm":
            from slam_llm.models.encoder import WavLMEncoder
            encoder = WavLMEncoder.load(model_config)
        if encoder_name == "av_hubert":
            from slam_llm.models.encoder import AVHubertEncoder
            encoder = AVHubertEncoder.load(model_config)
        if encoder_name == "hubert":
            from slam_llm.models.encoder import HubertEncoder
            encoder = HubertEncoder.load(model_config)
        if encoder_name == "musicfm":
            from slam_llm.models.encoder import MusicFMEncoder
            encoder = MusicFMEncoder.load(model_config)

        if "llama" in encoder_name.lower():
            from slam_llm.models.encoder import HfTextEncoder
            encoder = HfTextEncoder.load(model_config)
    print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters(): 
            param.requires_grad = False
        encoder.eval()
    print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    return encoder

def setup_llm(train_config, model_config, **kwargs):
    from pkg_resources import packaging
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None
    if (train_config.enable_fsdp or train_config.enable_ddp) and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        # v = packaging.version.parse(torch.__version__)
        # verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        # if not verify_latest_nightly:
        #     raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
        #                     "please install latest nightly.")
        rank = int(os.environ["RANK"])
        if rank == 0:
            if "vallex" in model_config.llm_name.lower():
                from src.slam_llm.models.vallex.vallex_config import VallexConfig
                from src.slam_llm.models.vallex.vallex_model import VALLE
                vallex_config = VallexConfig(
                    **model_config
                )
                model = VALLE(vallex_config)
            elif "aya" in model_config.llm_name.lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.llm_path,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.llm_path,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                    attn_implementation="flash_attention_2" if train_config.use_fast_kernels else None,
                    torch_dtype=torch.bfloat16
                )
        else:
            llama_config = AutoConfig.from_pretrained(model_config.llm_path)
            llama_config.use_cache = use_cache
            # with torch.device("meta"):
            if "aya" in model_config.llm_name.lower():
                model = AutoModelForSeq2SeqLM(llama_config)
            else:
                model = AutoModelForCausalLM(llama_config) #(FIX:MZY): torch 2.0.1 does not support `meta`

    else:
        if "vallex" in model_config.llm_name.lower():
            from src.slam_llm.models.vallex.vallex_config import VallexConfig
            from src.slam_llm.models.vallex.vallex_model import VALLE
            vallex_config = VallexConfig(
                **model_config
            )
            model = VALLE(vallex_config)
        elif "aya" in model_config.llm_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="flash_attention_2" if train_config.use_fast_kernels else None,
                torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="flash_attention_2" if train_config.use_fast_kernels else None,
                torch_dtype=torch.bfloat16
            )

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.freeze_llm: # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()
        
    if kwargs.get("peft_ckpt", None): # (FIX:MZY):reload will get wrong results when decoding
        logger.info("loading peft_ckpt from: {}".format(kwargs.get("peft_ckpt")))
        model = PeftModel.from_pretrained(model=model, model_id=kwargs.get("peft_ckpt"), is_trainable=True)
        model.print_trainable_parameters()
    elif train_config.use_peft:
        logger.info("setup peft...")
        peft_config = generate_peft_config(train_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model

def setup_encoder_projector(train_config, model_config, **kwargs):
    if model_config.encoder_projector == "linear":
        from slam_llm.models.projector import EncoderProjectorConcat
        encoder_projector = EncoderProjectorConcat(model_config)
    elif model_config.encoder_projector == "cov1d-linear":
        from slam_llm.models.projector import EncoderProjectorCov1d
        encoder_projector = EncoderProjectorCov1d(model_config)
    elif model_config.encoder_projector == "q-former":
        from slam_llm.models.projector_cl import EncoderProjectorQFormer
        encoder_projector = EncoderProjectorQFormer(model_config)
    else:
        return None
    print_module_size(encoder_projector, model_config.encoder_projector, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return encoder_projector


class slam_model(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        tokenizer, 
        train_config, 
        model_config, 
        **kwargs
    ):
        super().__init__()
        # modality encoder 
        self.encoder = encoder

        # llm
        self.llm = llm
        self.llm.gradient_checkpointing_enable()
        self.encoder.gradient_checkpointing_enable()

        P_global = torch.from_numpy(np.load("/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/scripts/Whisper_P_global_k4.npy"))
        self.projector = LSARProjector(P_global, freeze=True)   # ← 新增
        self.apply_lsar_at_infer = True  # 推理时是否启用

        # projector
        self.encoder_projector = encoder_projector
        # 
        # self.align_head = AlignmentHead(768)

        # dim = 768
        # self.q_proj = nn.Linear(dim, dim, bias=False)
        # self.k_proj = nn.Linear(dim, dim, bias=False)
        # self.scale  = dim ** -0.5
        # proj_dim = 256
        # self.sim_projector = nn.Sequential(
        #     nn.Linear(model_config.llm_dim, model_config.llm_dim, bias=False),
        #     _sync_bn(model_config.llm_dim),            # ← SyncBN
        #     nn.ReLU(inplace=True),
        #     nn.Linear(model_config.llm_dim, proj_dim, bias=False),
        # )

        # self.sim_predictor = nn.Sequential(
        #     nn.Linear(proj_dim, 256, bias=False),
        #     _sync_bn(256),                        # ← SyncBN
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, proj_dim)              # 原版输出无 BN/ReLU，如要 BN 可再加
        # )

        # nn.LayerNorm(proj_dim),
        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

        if train_config.get("enable_deepspeed", False):
            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)
            for item in self.modules():
                if isinstance(item, nn.LayerNorm):
                    item.forward = types.MethodType(new_forward, item)



    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        audio_mel = kwargs.get("audio_mel", None)

        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # 2x downsample for whisper

        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)
        visual = kwargs.get("visual", None)
        visual_mask = kwargs.get("visual_mask", None)


        # for text encoder
        instruct_ids = kwargs.get("instruct_ids", None)
        instruct_mask = kwargs.get("instruct_mask", None)

        
        encoder_outs = None
        if audio_mel is not None or audio is not None or visual is not None:
            if self.train_config.freeze_encoder: # freeze encoder
                self.encoder.eval()
            if self.model_config.encoder_path_hf is not None:
                encoder_outs = self.encoder(audio_mel.permute(0, 2, 1)).last_hidden_state # bs*seq*dim
            elif self.model_config.encoder_name == "whisper":
                encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1)) # bs*seq*dim
            if self.model_config.encoder_name == "beats":
                encoder_outs, audio_mel_post_mask = self.encoder.extract_features(audio_mel, audio_mel_mask) # bs*seq*dim
            if self.model_config.encoder_name == "eat":
                encoder_outs = self.encoder.model.extract_features(audio_mel.unsqueeze(dim=1), padding_mask = None, mask=False, remove_extra_tokens = False)['x']
            if self.model_config.encoder_name == "SpatialAST":
                encoder_outs = self.encoder(audio) # output: [bs, seq_len=3+512, dim=768]
            if self.model_config.encoder_name == "wavlm":
                encoder_outs = self.encoder.extract_features(audio, 1 - audio_mask) #(FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
            if self.model_config.encoder_name == "hubert":
                results = self.encoder(source = audio, padding_mask = 1-audio_mask)
                if self.model_config.encoder_type == "pretrain":
                    encoder_outs, audio_mel_post_mask = results["x"], results["padding_mask"]
                if self.model_config.encoder_type == "finetune":
                    encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                    encoder_outs = encoder_outs.transpose(0, 1)
            if self.model_config.encoder_name == "av_hubert":
                results = self.encoder(source={'video':visual, 'audio':audio}, padding_mask=visual_mask) # bs*seq*dim  
                encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                encoder_outs = encoder_outs.transpose(0, 1)
                audio_mel_post_mask = (~audio_mel_post_mask).float()
            if self.model_config.encoder_name == 'musicfm':
                encoder_outs = self.encoder.extract_features(audio, padding_mask = None) # MusicFM doesn't support padding mask 
            if self.encoder is None:
                encoder_outs = audio_mel if audio_mel is not None else audio

            if self.training or self.apply_lsar_at_infer:
                encoder_outs = self.projector(encoder_outs)

            if self.training and self.model_config.encoder_projector == "q-former":
                encoder_outs, shallow_query = self.encoder_projector(encoder_outs, audio_mel_post_mask,1)
            if not self.training and self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
            if self.model_config.encoder_projector == "cov1d-linear": 
                encoder_outs = self.encoder_projector(encoder_outs) 

        if instruct_ids is not None:
            if self.encoder is not None:
                encoder_outs = self.encoder(input_ids=instruct_ids, attention_mask=instruct_mask).last_hidden_state

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, instruct_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        
        # loss_w = None
        # loss_align = None
        
        # if self.training and ("view1" in kwargs) and ("view2" in kwargs):
        #     loss_w = torch.tensor(0.0, device=encoder_outs.device)
        #     loss_align = torch.tensor(0.0, device=encoder_outs.device)

        #     # for name, param in self.align_head.named_parameters():
        #     #     print(f"Param: {name}, requires_grad={param.requires_grad}")
        #     # if self.align_head.q_proj.weight.grad is not None:
        #     #     grad_mean = self.align_head.q_proj.weight.grad.abs().mean().item()
        #     #     print(f"q_proj weight grad mean: {grad_mean:.6f}")
        #     # else:
        #     #     print("q_proj.weight grad is None!")
        #     # print("shallow_query requires_grad:", shallow_query.requires_grad)

        #     for i in range(0, shallow_query.size(0), 2):
        #         en_query = shallow_query[i]
        #         ja_query = shallow_query[i+1]
        #         en_query = F.normalize(en_query, dim=-1)  # [Q, D]
        #         ja_query = F.normalize(ja_query, dim=-1)  # [Q, D]

        #         loss_w += sinkhorn_wasserstein_loss(ja_query, en_query)
        #         z_hat = self.align_head(ja_query.unsqueeze(0), en_query.unsqueeze(0))  # [1, Q, Q]
        #         # print("align_out requires_grad:", z_hat.requires_grad)

        #         # with torch.no_grad():
        #         cost = torch.cdist(ja_query, en_query, p=2) ** 2
        #         plan = gumbel_sinkhorn_sample(cost, temperature=0.2)
        #         plan = plan / plan.sum(dim=-1, keepdim=True)      

        #         loss_align += F.kl_div(z_hat.squeeze(0).log(), plan, reduction="batchmean")

        #         # print("loss_align:", loss_align.item())
        #         # print("z_hat mean:", z_hat.mean().item())
        #         # print("plan mean:", plan.mean().item())
        #         # print("kl diff:", (z_hat.squeeze(0).log() - plan.log()).abs().mean().item())
        #         # print("plan min:", plan.min().item(), "max:", plan.max().item())
        #         # assert False
        #     # z_hat.retain_grad()  
        #     # print("z_hat grad mean =", z_hat.grad.abs().mean().item())

        #     loss_w = loss_w / (shallow_query.size(0) // 2)
        #     loss_align = loss_align / (shallow_query.size(0) // 2)

            # print(loss_w)
            # print(loss_align)


        loss_w = None
        loss_span = None
        if self.training and ("view1" in kwargs) and ("view2" in kwargs):
            loss_w = torch.tensor(0.0, device=encoder_outs.device)
            loss_span = torch.tensor(0.0, device=encoder_outs.device)
            # loss_w2 = torch.tensor(0.0, device=encoder_outs.device)
            for i in range(0, shallow_query.size(0), 2):
                en_query = shallow_query[i]     # [Q, D]
                x_query  = shallow_query[i+1]   # [Q, D]
                en_query = F.normalize(en_query, dim=-1)
                x_query = F.normalize(x_query, dim=-1)

                loss_w += 0.5 * sinkhorn_wasserstein_loss(x_query, en_query)

                span_en = span_pooling(en_query, w=4, stride=2, mode="mean")  # [M, D]
                span_x  = span_pooling(x_query,  w=4, stride=2, mode="mean")  # [M, D]
                span_en = F.normalize(span_en, dim=-1)
                span_x  = F.normalize(span_x, dim=-1)

                loss_span += 0.5 * sinkhorn_wasserstein_loss(span_x, span_en)

            # loss_w = loss_w / (shallow_query.size(0) // 2)  # 平均化
            # loss_span = loss_span / (shallow_query.size(0) // 2)

        # print(loss_span)
        # assert False

        # loss_align = None
        # if self.training and ("view1" in kwargs) and ("view2" in kwargs):
        #     B_pairs = shallow_query.size(0) // 2
        #     kl_sum  = 0.0
        #     for i in range(0, shallow_query.size(0), 2):

        #         en_tok = F.normalize(shallow_query[i],   dim=-1).unsqueeze(0)  # [1,Q,D]
        #         ja_tok = F.normalize(shallow_query[i+1], dim=-1).unsqueeze(0)  # [1,Q,D]

        #         z_hat = self.align_head(ja_tok, en_tok)         

        #         with torch.no_grad():
        #             C = torch.cdist(ja_tok[0], en_tok[0], p=2)   # [Q,Q]
        #             C = C.to(dtype=torch.float32)               
        #             Z_ot = torch.from_numpy(
        #                 ot.emd2([], [], C.cpu().numpy(), log=True)[1]['G']
        #             ).to(C.device)                      # [Q,Q]
        #             Z_ot = Z_ot / Z_ot.sum(dim=-1, keepdim=True)

        #         kl_sum += F.kl_div(z_hat.log(), Z_ot, reduction="batchmean")

        #     loss_align = kl_sum / B_pairs

      

            # for i in range(0, encoder_outs.size(0), 2):
            #     en_query2 = encoder_outs[i]     # [Q, D]
            #     x_query2  = encoder_outs[i+1]   # [Q, D]
            #     en_query2 = F.normalize(en_query2, dim=-1)
            #     x_query2 = F.normalize(x_query2, dim=-1)

            #     loss_w2 +=   * sinkhorn_wasserstein_loss(x_query2, en_query2)

            # loss_w = loss_w / (shallow_query.size(0) // 2)  # 平均化
            # loss_w2 = loss_w2 / (encoder_outs.size(0) // 2)

            # for i in range(0, shallow_query2.size(0), 2):
            #     en_query2 = shallow_query2[i]     # [Q, D]
            #     x_query2  = shallow_query2[i+1]   # [Q, D]
            #     en_query2 = F.normalize(en_query2, dim=-1)
            #     x_query2 = F.normalize(x_query2, dim=-1)

            #     loss_w2 += 0.5 * sinkhorn_wasserstein_loss(x_query2, en_query2)

            # loss_w = loss_w / (shallow_query.size(0) // 2)  # 平均化
            # loss_w2 = loss_w2 / (shallow_query2.size(0) // 2)
            # print(loss_ss)
            # assert False
        # loss_ss = None
        # if self.training and ("view1" in kwargs) and ("view2" in kwargs):
        #     loss_ss = torch.tensor(0.0, device=encoder_outs.device)

        #     sent_vec = encoder_outs.mean(dim=1)

        #     en_vec = sent_vec[0::2]                    # [B, D]
        #     x_vec = sent_vec[1::2]

        #     loss_ss = F.mse_loss(x_vec, en_vec.detach())
        # loss_ss = None
        # if self.training and ("view1" in kwargs) and ("view2" in kwargs):

        #     # 假设 batch: en, fr, en, fr, ...
        #     en_tok = encoder_outs[0::2]          # [B/2, T, D]
        #     x_tok = encoder_outs[1::2]          # 法语或日语

        #     # 单位化
        #     en_norm = F.normalize(en_tok, dim=-1)
        #     x_norm = F.normalize(x_tok, dim=-1)

        #     # 逐 token 余弦相似度
        #     cos_sim = (x_norm * en_norm.detach()).sum(dim=-1)   # [B/2, T]

        #     # 损失 = 1 - cosine
        #     tok_loss = 1.0 - cos_sim        # 越小越好

        #     loss_ss = tok_loss.mean()
            # print(loss_ss)
            # assert False


            # z      = self.sim_projector(sent_vec)            # [B, d]
            # z1, z2 = z[0::2], z[1::2]
            # p1, p2 = self.sim_predictor(z1), self.sim_predictor(z2)
            # def pos_euclid_loss(p, z):
            #     # 直接用 L2 距离；也可改成平方距离
            #     return F.pairwise_distance(p, z.detach(), p=2).mean()

            # loss_ss = pos_euclid_loss(p2, z1.detach())

            # print(loss_ss)
        
        # tau = 0.07          # 温度，可自行调

        # loss_ss = None
        # if self.training and ("view1" in kwargs) and ("view2" in kwargs):
        #     # 1) 直接对 QFormer 输出做 mean-pool 得句向量
        #     sent_vec = encoder_outs.mean(dim=1)                 # [B, hidden_dim]

        #     # 2) 归一化后作为特征 z
        #     z = F.normalize(sent_vec, dim=-1)                   # [B, hidden_dim]

        #     z1, z2 = z[0::2], z[1::2]                           # [B/2, hidden_dim] ×2
        #     B = z1.size(0)

        #     feats = torch.cat([z1, z2], dim=0)                  # [2B, hidden_dim]
        #     sim   = torch.mm(feats, feats.t()) / tau            # 余弦相似 / τ

        #     diag_mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
        #     sim.masked_fill_(diag_mask, -1e9)

        #     positives = torch.cat([
        #         torch.arange(B, 2 * B, device=sim.device),
        #         torch.arange(0, B, device=sim.device)
        #     ])

        #     loss_ss = F.cross_entropy(sim, positives)
        # tau = 0.07
        # loss_ss = None
        # if self.training and ("view1" in kwargs) and ("view2" in kwargs):
        # # encoder_outs:[2B,hidden_dim]
        #     sent_vec = encoder_outs.mean(dim=1)       # [2B, D]
        #     # 不做归一化，直接用原始向量
        #     z = sent_vec                              # [2B, D]

        #     # 偶数位置是 English，奇数位置是 Japanese
        #     z1 = z[0::2]   # [B, D]
        #     z2 = z[1::2]   # [B, D]
        #     B  = z1.size(0)

        #     d_pos = F.pairwise_distance(z1, z2, p=2)   # [B]

        #     # 负例示例：把 z2 整体向下滚动一位（每个 z1_i 对应 z2_{(i+1)%B}）
        #     z2_neg = torch.roll(z2, shifts=1, dims=0)  # [B, D]
        #     d_neg = F.pairwise_distance(z1, z2_neg, p=2)  # [B]

        #     # margin 超参
        #     margin = 1.0
        #     # 两元对比损失：正例越近越好，负例距离大于 margin
        #     loss_pos = 0.5 * (d_pos**2).mean()
        #     loss_neg = 0.5 * F.relu(margin - d_neg).pow(2).mean()
        #     loss_ss = loss_pos + loss_neg

        #     print(loss_ss)
            
        #     assert False

        # if torch.isnan(loss_ss) or torch.isinf(loss_ss):
        #     raise ValueError("loss_ss 为 NaN/Inf，检查 sim 或 positives 构造是否正确")

        # with torch.no_grad():
        #     B_pairs = z1.size(0)                           
        #     pos_sim = F.cosine_similarity(z1, z2, dim=-1).mean()
        #     neg_sim = F.cosine_similarity(z1, z1.roll(1, 0), dim=-1).mean()
        #     print(f"[Sanity] loss={loss_ss.item():.4f}  pos_sim={pos_sim.item():.3f}  "
        #         f"neg_sim={neg_sim.item():.3f}")
        # assert False

        input_ids = input_ids[:, 80:]


        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        inputs_embeds = torch.cat((encoder_outs, inputs_embeds), dim=1)
        
        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask


        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,)
        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(input=model_outputs.logits, dim=-1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)

        if loss_w is not None and loss_span is not None:
            return model_outputs, acc, loss_w, loss_span
        else: 
            return model_outputs, acc
    
    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            # max_length=kwargs.get("max_length", 200),
            max_new_tokens=kwargs.get("max_new_tokens", 150),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            no_repeat_ngram_size=4,
            early_stopping=True,
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return model_outputs