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
import random

from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.train_utils import print_module_size, print_model_size
from peft import PeftModel, PeftConfig
from torch.nn import CrossEntropyLoss
from slam_llm.utils.metric import compute_accuracy
from geomloss import SamplesLoss
import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

# LANG2ID = {
#     "fleurs_en_us": 0,
#     "fleurs_ja_jp": 1,
#     "fleurs_es_419": 2,
#     "fleurs_ko_kr": 3,
#     "fleurs_ru_ru": 4
# }
# LANG2ID = {
#     "fleurs_en_us": 0,
#     "fleurs_ja_jp": 1,
#     "fleurs_fr_fr": 2,
#     "fleurs_ko_kr": 3,
#     "fleurs_de_de": 4
# }
LANG2ID = {
    "fleurs_en_us": 0,
    "fleurs_ja_jp": 1,
    "fleurs_yue_hant_hk": 2,
    "fleurs_ko_kr": 3,
    "fleurs_it_it": 4
}

# class DynamicBias(nn.Module):
#     def __init__(self, centroids: torch.Tensor, hidden_size=1280, gate_dim='scalar'):
#         super().__init__()
#         self.bias_table = nn.Parameter(centroids.clone())         # [L, D]
#         self.ln   = nn.LayerNorm(hidden_size)
#         self.mlp  = nn.Sequential(
#             nn.Linear(hidden_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1 if gate_dim=='scalar' else hidden_size),
#             nn.Sigmoid()           # 输出 α∈(0,1)
#         )
#         self.gate_dim = gate_dim

#     def forward(self, h: torch.Tensor, lang_id: torch.LongTensor):

#             sent = h.mean(1)                    # [B, D]
#             alpha = self.mlp(self.ln(sent))     # [B, 1] or [B, D]

#             b = self.bias_table[lang_id]        # [B, D]

#             delta = alpha * b
#             h = h + delta.unsqueeze(1)          
#             return h, alpha.detach()                 

sinkhorn = SamplesLoss(loss="sinkhorn",p=2,blur=0.05)
def sinkhorn_wasserstein_loss(x_seq, y_seq):
    return sinkhorn(x_seq, y_seq)

class LayerSampler:
    def __init__(
        self,
        candidates: Optional[List[int]] = None,
        strategy: str = "uniform",
        weights: Optional[List[float]] = None,
        device: torch.device = torch.device("cpu"),
        alpha: float = 0.9,                 # EMA
        beta:  float = 1,                 # UCB
    ):
        self.device = device
        self.candidates = candidates
        self.strategy = strategy
        self.weights  = torch.tensor(
            weights if weights is not None else [1.0] * len(self.candidates),
            dtype=torch.float,
            device=device,
        )
        # UCB
        
        self.alpha  = alpha
        self.beta   = beta
        self.Q      = torch.zeros(len(candidates), device=device)   # EMA Δ-Cost 奖励
        self.count  = torch.zeros(len(candidates), device=device)   # 被选次数
        self.prev_cost = [None] * len(candidates)                   # 上次 cost
        self.total_t   = 0                            # 总选择次数

    def sample(self) -> int:
        if self.strategy == "uniform":
            idx = random.choice(self.candidates)
        elif self.strategy == "weighted":
            probs = self.weights / self.weights.sum()
            # idx = torch.multinomial(probs, 1).item() + 1

            rel   = torch.multinomial(probs, 1).item()
            idx   = self.candidates[rel] + 1

        elif self.strategy == "ucb_bandit":
            # UCB: score = Q - β * sqrt(log t / n)

            self.total_t += 1
            # if self.total_t <= 3500:
            #     idx = random.choice(self.candidates) + 1
            # else:
            expl = self.beta * torch.sqrt(
                torch.log(torch.tensor(self.total_t, device=self.device)) /
                (self.count + 1e-6)
            )
            ucb_scores = self.Q - expl
            # rel = torch.argmax(ucb_scores).item()
            # idx = self.candidates[rel]
            temp = max(0.05, 1.0 * (0.999 ** self.total_t)) 
            probs = torch.softmax(-ucb_scores / temp, dim=0)

            rel = torch.multinomial(probs, num_samples=1).item()
            idx = self.candidates[rel]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return idx

    def update_weights(self, new_scores: List[float], tau: float = 1.0):

        scores = torch.tensor(new_scores, dtype=torch.float, device=self.device)
        self.weights = torch.softmax(scores / tau, dim=-1)

    def update_ucb(self, layer_idx: int, cost_now: float):
        rel = self.candidates.index(layer_idx)

        if self.prev_cost[rel] is not None:
            delta = self.prev_cost[rel] - cost_now
            self.Q[rel] = self.alpha * self.Q[rel] + (1 - self.alpha) * delta

        self.prev_cost[rel] = cost_now
        self.count[rel]    += 1
        

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


        sampler = LayerSampler(
            candidates=[1,2,3],  # e.g. [0,1,2,3,4]
            strategy="ucb_bandit"
        )
        self.ot_sampler = sampler
        # self.ema_loss = [None] * len([1,2,3])

        # modality encoder 
        self.encoder = encoder

        centroids = torch.tensor(np.load("/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/scripts/bias_enjpyuekoit_largev3.npy"), dtype=torch.float32)
        self.bias_table = nn.Parameter(centroids.clone(), requires_grad=True)
        # self.scale_table = nn.Parameter(torch.ones_like(self.bias_table))
        # self.db = DynamicBias(centroids)

        # llm
        self.llm = llm
        self.llm.gradient_checkpointing_enable()
        self.encoder.gradient_checkpointing_enable()
        
        # projector
        self.encoder_projector = encoder_projector
      
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
        sources = kwargs.get("sources", None)
        # print(sources)
        # if not self.training: 
        #     assert False
        
        encoder_outs = None
        if audio_mel is not None or audio is not None or visual is not None:
            if self.train_config.freeze_encoder: # freeze encoder
                self.encoder.eval()
            if self.model_config.encoder_path_hf is not None:
                encoder_outs = self.encoder(audio_mel.permute(0, 2, 1)).last_hidden_state # bs*seq*dim
                lang_id = torch.tensor([LANG2ID[s] for s in sources],
                                    dtype=torch.long)

                b = self.bias_table[lang_id]                # [B, 1280]

                encoder_outs = encoder_outs + b.unsqueeze(1)

                # encoder_out, _ = self.db(encoder_outs, lang_id)
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
            if self.training and self.model_config.encoder_projector == "q-former":
                ot_layer = torch.randint(
                    low=1,
                    high=4,
                    size=(1,),
                    device=encoder_outs.device,
                ).item()
                ot_layer = self.ot_sampler.sample()
                # ot_layer = 1
                encoder_outs, shallow_query = self.encoder_projector(encoder_outs, audio_mel_post_mask, ot_layer)
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


        loss_w = None
        if self.training and ("view1" in kwargs) and ("view2" in kwargs):
            loss_w = torch.tensor(0.0, device=encoder_outs.device)
            B = shallow_query.size(0)

            for i in range(0, B, 2):
                en_query = shallow_query[i]     # [Q, D]
                x_query  = shallow_query[i+1]   # [Q, D]
                en_query = F.normalize(en_query, dim=-1)
                x_query = F.normalize(x_query, dim=-1)

                loss_w += 0.5 * sinkhorn_wasserstein_loss(x_query, en_query)
            

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

        # if loss_w is not None and loss_span is not None:
        #     return model_outputs, acc, loss_w, loss_span
        if loss_w is not None:
            return model_outputs, acc, loss_w, ot_layer
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