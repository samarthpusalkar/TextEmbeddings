import os
import torch
from transformers import BertModel, BertConfig, BertForTokenClassification
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class CustomBertConfig(BertConfig):
    def __init__(self, global_layer_index=5, global_dim=128, seq_length=512, num_labels=30522, **kwargs):
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

class GlobalDenseAggregator(torch.nn.Module):
    def __init__(self, seq_length, hidden_size, global_dim):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.global_dim = global_dim
        # self.aggregator = torch.nn.Linear(seq_length * hidden_size, global_dim)
        self.aggregator = torch.nn.Linear(hidden_size, global_dim)
        self.upscale_decoder = torch.nn.Linear(global_dim, seq_length * hidden_size)
        self.activation = torch.nn.LeakyReLU()
    
    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_length, hidden_size)
        batch_size, seq_length, hidden_size = hidden_states.size()
        # assert seq_length == self.seq_length, f"Expected seq_length {self.seq_length}, got {seq_length}"
        # flat = hidden_states.view(batch_size, -1)  # (batch_size, seq_length * hidden_size)
        # global_vector = self.activation(self.aggregator(flat))  # (batch_size, global_dim)
        cls_token = hidden_states[:,0,:]# (batch_size, seq_length * hidden_size)
        global_vector = self.activation(self.aggregator(cls_token))  # (batch_size, global_dim)
        upscale_decode = self.activation(self.upscale_decoder(global_vector))  # (batch_size, seq_length * hidden_size)
        new_hidden_states = upscale_decode.view(batch_size, seq_length, hidden_size)
        # Inject global information into each token (here via addition).
        return new_hidden_states

from transformers.models.bert.modeling_bert import BertEncoder, BertLayer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, TokenClassifierOutput

class CustomBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.global_layer_index = config.global_layer_index
        self.global_dense = GlobalDenseAggregator(seq_length=config.seq_length,
                                                  hidden_size=config.hidden_size,
                                                  global_dim=config.global_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False


        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            
            # Insert global aggregator after the specified layer.
            if i == self.global_layer_index:
                hidden_states = self.global_dense(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertSentenceEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.global_layer_index = config.num_hidden_layers
        self.global_dense = GlobalDenseAggregator(seq_length=config.seq_length,
                                                  hidden_size=config.hidden_size,
                                                  global_dim=config.global_dim)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        encodings = self.global_dense(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertForSentenceEncodingsTraining(BertForTokenClassification):
    # Tell Transformers to use our custom config class.
    config_class = CustomBertConfig

    def __init__(self, config):
        super().__init__(config)
        # Replace the bert Encoder with Custom Encoder
        self.bert.encoder = CustomBertEncoder(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output = outputs.logits  # (batch_size, seq_length, hidden_size)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if input_ids is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), input_ids.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
        num_labels = base_config.vocab_size
        base_config.num_labels = num_labels
        for key in ["global_layer_index", "global_dim", "seq_length"]:
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
        
    def get_encoder(self):
        pass
