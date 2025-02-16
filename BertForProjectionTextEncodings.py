import os
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from typing import Optional, Union

# 1. Custom configuration that stores extra parameters.
class CustomBertConfig(BertConfig):
    def __init__(self, global_layer_index=5, global_dim=128, seq_length=512, **kwargs):
        super().__init__(**kwargs)
        self.global_layer_index = global_layer_index  # After which layer to inject our custom layer.
        self.global_dim = global_dim                  # Dimension for our global vector.
        self.seq_length = seq_length                  # Fixed sequence length.
    
    def to_dict(self):
        output = super().to_dict()
        output["global_layer_index"] = self.global_layer_index
        output["global_dim"] = self.global_dim
        output["seq_length"] = self.seq_length
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict)
    
    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path: Union[str, os.PathLike],cache_dir: Optional[Union[str, os.PathLike]] = None,force_download: bool = False,local_files_only: bool = False,token: Optional[Union[str, bool]] = None,revision: str = "main",**kwargs):
        return BertConfig.from_pretrained(pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, **kwargs)


# 2. Custom global aggregator layer.
class GlobalDenseAggregator(nn.Module):
    def __init__(self, seq_length, hidden_size, global_dim):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.global_dim = global_dim
        self.aggregator = nn.Linear(seq_length * hidden_size, global_dim)
        self.injector = nn.Linear(global_dim, seq_length * hidden_size)
        self.activation = nn.ReLU()
    
    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_length, hidden_size)
        batch_size, seq_length, hidden_size = hidden_states.size()
        assert seq_length == self.seq_length, f"Expected seq_length {self.seq_length}, got {seq_length}"
        flat = hidden_states.view(batch_size, -1)  # (batch_size, seq_length * hidden_size)
        global_vector = self.activation(self.aggregator(flat))  # (batch_size, global_dim)
        injected_flat = self.activation(self.injector(global_vector))  # (batch_size, seq_length * hidden_size)
        injected = injected_flat.view(batch_size, seq_length, hidden_size)
        # Inject global information into each token (here via addition).
        return hidden_states + injected

# 3. Custom encoder that inserts the global aggregator after a specific layer.
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class CustomBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.global_layer_index = config.global_layer_index
        self.global_dense = GlobalDenseAggregator(seq_length=config.seq_length,
                                                  hidden_size=config.hidden_size,
                                                  global_dim=config.global_dim)
        # Build the standard BERT layers.
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values[i] if past_key_values is not None else None,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            # Insert global aggregator after the specified layer.
            if i == self.global_layer_index:
                hidden_states = self.global_dense(hidden_states)
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return (hidden_states, all_hidden_states, all_attentions)
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            pooler_output=None,
        )

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path):
    #     return super().from_pretrained(path)

# 4. Custom model for token classification.
class CustomBertForTokenClassification(BertPreTrainedModel):
    # Tell Transformers to use our custom config class.
    config_class = CustomBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Load the base BERT model and replace its encoder with our custom encoder.
        self.bert = BertModel(config)
        self.bert.encoder = CustomBertEncoder(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits) if loss is not None else logits

    @classmethod
    def from_base_for_training(cls, pretrained_model_name_or_path, **kwargs):
        """
        Custom initialization for training.
        1. Loads the base BERT config.
        2. Injects custom parameters (global_layer_index, global_dim, seq_length).
        3. Instantiates the custom model.
        4. Loads matching pretrained weights into the BERT parts.
        """
        # Load base config.
        base_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        # Update with custom parameters.
        for key in ["global_layer_index", "global_dim", "seq_length", "num_labels"]:
            if key in kwargs:
                setattr(base_config, key, kwargs[key])
                kwargs.pop(key)
        # Create a custom config.
        custom_config = CustomBertConfig.from_dict(base_config.to_dict())
        # Instantiate the model.
        model = cls(custom_config)
        
        # Load pretrained weights for BERT parts.
        pretrained_bert = BertModel.from_pretrained(pretrained_model_name_or_path, config=base_config, **kwargs)
        pretrained_state_dict = pretrained_bert.state_dict()
        model_bert_state_dict = model.bert.state_dict()
        # Only load matching keys.
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_bert_state_dict}
        model.bert.load_state_dict(filtered_state_dict, strict=False)
        
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Custom from_pretrained that loads the config and then the state dict
        from either a PyTorch or SafeTensors file, depending on the flag
        `use_safetensors` passed via kwargs.
        """
        model = super().from_pretrained(pretrained_model_name_or_path)
        return model

    def save_pretrained(self, save_directory, **kwargs):
        """
        Override save_pretrained if needed. The default implementation saves the model's state_dict
        and the config (via config.to_dict()), which already includes our custom parameters.
        """
        # You can add custom saving behavior here if necessary.
        super().save_pretrained(save_directory, **kwargs)
