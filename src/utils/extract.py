import os




def extract_config(model_type, model):
    
    if model_type == "bert":
        model_config = {
            "d_emb": model.config.hidden_size,
            "d_ffn": model.config.intermediate_size,
            "head_num": model.config.num_attention_heads,
            "head_size": model.config.hidden_size // model.config.num_attention_heads,
            "layer_num": model.config.num_hidden_layers,
        }
    elif model_type == "gpt2":
        model_config = {
            "d_emb": model.config.n_embd,
            "d_ffn": model.config.n_embd * 4,
            "head_num": model.config.n_head,
            "head_size": model.config.n_embd // model.config.n_head,
            "layer_num": model.config.n_layer,
        }

    return model_config



def extract_key(model_type):
    if model_type == "bert":
        model_key = {
            "prefix": "encoder",
            "layer_key": "layer",
            "attn_q_key": "attention.self.query",
            "attn_k_key": "attention.self.key",
            "attn_v_key": "attention.self.value",
            "attn_o_key": "attention.output.dense",
            "ffn1_key": "intermediate.dense",
            "ffn2_key": "output.dense",
        }
    elif model_type == "bert_classifier":
        model_key = {
            "prefix": "bert.encoder",
            "layer_key": "layer",
            "attn_key": "attention",
            "attn_q_key": "attention.self.query",
            "attn_k_key": "attention.self.key",
            "attn_v_key": "attention.self.value",
            "attn_o_key": "attention.output.dense",
            "ffn1_key": "intermediate.dense",
            "ffn2_key": "output.dense",
        }
    elif model_type == "gpt2":
        model_key = {
            "prefix": "",
            "layer_key": "h",
            "attn_key": "attn",
            "attn_o_key": "attn.c_proj",
            "ffn1_key": "mlp.c_fc",
            "ffn2_key": "mlp.c_proj",
        }
    elif model_type == "gpt2_lm":
        model_key = {
            "prefix": "transformer",
            "layer_key": "h",
            "attn_key": "attn",
            "attn_o_key": "attn.c_proj",
            "ffn1_key": "mlp.c_fc",
            "ffn2_key": "mlp.c_proj",
        }
    return model_key


