import torch
from transformers import AutoModel
import pdb
from transformers.pytorch_utils import Conv1D


class Architecture():
    def __init__(
        self,
        model,
        model_key,
        model_config,
    ):
        self.model = model
        self.model_config = model_config
        self.model_key = model_key
        
        self.prefix = model_key["prefix"] if "prefix" in model_key.keys() else None
        self.layer_key = model_key["layer_key"] if "layer_key" in model_key.keys() else None
        self.attn_key = model_key["attn_key"] if "attn_key" in model_key.keys() else None
        self.attn_q_key = model_key["attn_q_key"] if "attn_q_key" in model_key.keys() else None
        self.attn_k_key = model_key["attn_k_key"] if "attn_k_key" in model_key.keys() else None
        self.attn_v_key = model_key["attn_v_key"] if "attn_v_key" in model_key.keys() else None
        self.attn_o_key = model_key["attn_o_key"] if "attn_o_key" in model_key.keys() else None
        self.ffn1_key = model_key["ffn1_key"] if "ffn1_key" in model_key.keys() else None
        self.ffn2_key = model_key["ffn2_key"] if "ffn2_key" in model_key.keys() else None
    
    def _set_module(self, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = self.model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)

    def get_encoder(self):
        assert self.prefix is not None
        encoder = self.model
        if len(self.prefix) > 0:
            for temp_key in self.prefix.split("."):
                encoder = encoder.__getattr__(temp_key)
        else:
            encoder = self.model
        return encoder
    
    def get_layer(self, layer_index):
        encoder = self.get_encoder()
        layer = encoder.__getattr__(self.layer_key)[layer_index]
        return layer
    
    def get_attn(self, layer_index):
        assert self.attn_key is not None
        attn = self.get_layer(layer_index)
        attn_key = self.attn_key
        for temp_key in attn_key.split("."):
            attn = attn.__getattr__(temp_key)
        return attn
    
    def get_ffn1(self, layer_index):
        assert self.ffn1_key is not None
        ffn1 = self.get_layer(layer_index)
        for temp_key in self.ffn1_key.split("."):
            ffn1 = ffn1.__getattr__(temp_key)
        return ffn1
    
    def get_ffn2(self, layer_index):
        assert self.ffn2_key is not None
        ffn2 = self.get_layer(layer_index)
        for temp_key in self.ffn2_key.split("."):
            ffn2 = ffn2.__getattr__(temp_key)
        return ffn2

    def get_attn_q(self, layer_index):
        assert self.attn_q_key is not None
        attn_q = self.get_layer(layer_index)
        for temp_key in self.attn_q_key.split("."):
            attn_q = attn_q.__getattr__(temp_key)
        return attn_q

    def get_attn_k(self, layer_index):
        assert self.attn_k_key is not None
        attn_k = self.get_layer(layer_index)
        for temp_key in self.attn_k_key.split("."):
            attn_k = attn_k.__getattr__(temp_key)
        return attn_k

    def get_attn_v(self, layer_index):
        assert self.attn_v_key is not None
        attn_v = self.get_layer(layer_index)
        for temp_key in self.attn_v_key.split("."):
            attn_v = attn_v.__getattr__(temp_key)
        return attn_v
    
    def get_attn_o(self, layer_index):
        assert self.attn_o_key is not None
        attn_o = self.get_layer(layer_index)
        for temp_key in self.attn_o_key.split("."):
            attn_o = attn_o.__getattr__(temp_key)
        return attn_o

    def prune_attn(self, layer_index, head_mask):
        attn = self.get_attn(layer_index)
        attn.prune_heads(torch.where(head_mask==0)[0])
        
    
    def prune_ffn(self, layer_index, neuron_mask):
        d_emb = self.model_config["d_emb"]
        d_pruned = int(torch.sum(neuron_mask))
        unmask_index = torch.where(neuron_mask==1)[0]
        
        # ffn1
        state = self.get_ffn1(layer_index).state_dict()
        if isinstance(self.get_ffn1(layer_index), torch.nn.Linear):
            state["weight"] = state["weight"][unmask_index, :]
            state["bias"] = state["bias"][unmask_index]
            module = torch.nn.Linear(d_emb, d_pruned)
        elif isinstance(self.get_ffn1(layer_index), Conv1D):
            state["weight"] = state["weight"][:, unmask_index]
            state["bias"] = state["bias"][unmask_index]
            module = Conv1D(d_pruned, d_emb)
        module.load_state_dict(state)
        tokens = ".".join([self.prefix, self.layer_key, str(layer_index), self.ffn1_key])if len(self.prefix) > 0 else ".".join([self.layer_key, str(layer_index), self.ffn1_key])
        self._set_module(tokens, module)
        
        # ffn2
        state = self.get_ffn2(layer_index).state_dict()
        if isinstance(self.get_ffn2(layer_index), torch.nn.Linear):
            state["weight"] = state["weight"][:, unmask_index]
            module = torch.nn.Linear(d_pruned, d_emb)
        elif isinstance(self.get_ffn2(layer_index), Conv1D):
            state["weight"] = state["weight"][unmask_index, :]
            module = Conv1D(d_emb, d_pruned)
        module.load_state_dict(state)
        tokens = ".".join([self.prefix, self.layer_key, str(layer_index), self.ffn2_key]) if len(self.prefix) > 0 else ".".join([self.layer_key, str(layer_index), self.ffn2_key])
        self._set_module(tokens, module)
    
    def prune(self, head_mask, neuron_mask):
        layer_num = self.model_config["layer_num"]
        for layer_idx in range(layer_num):
            self.prune_attn(layer_idx, head_mask[layer_idx])
            self.prune_ffn(layer_idx, neuron_mask[layer_idx])