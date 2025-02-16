import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class GlobalDenseAggregator(nn.Module):
    """
    Aggregates all token representations (with fixed seq_length) into a single global vector,
    then projects that vector back to token-level representations.
    
    Args:
        seq_length (int): Fixed sequence length (e.g. 512)
        hidden_size (int): Hidden size of BERT
        global_dim (int): Dimension of the global vector (N)
    """
    def __init__(self, seq_length, hidden_size, global_dim):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.global_dim = global_dim
        # Fully connect the flattened token representations to a global vector
        self.aggregator = nn.Linear(seq_length * hidden_size, global_dim)
        # Fully connect the global vector back to the flattened token representation
        self.injector = nn.Linear(global_dim, seq_length * hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_length, hidden_size)
        batch_size, seq_length, hidden_size = hidden_states.size()
        # Ensure the sequence length is as expected
        assert seq_length == self.seq_length, f"Expected seq_length {self.seq_length}, got {seq_length}"
        # Flatten tokens: (batch_size, seq_length * hidden_size)
        flat = hidden_states.view(batch_size, -1)
        # Get a global vector of shape (batch_size, global_dim)
        global_vector = self.activation(self.aggregator(flat))
        # Project back to token-level (flattened)
        injected_flat = self.activation(self.injector(global_vector))
        # Reshape to (batch_size, seq_length, hidden_size)
        injected = injected_flat.view(batch_size, seq_length, hidden_size)
        # Combine the injected global information with the original hidden states (here via addition)
        return hidden_states + injected

class CustomBertEncoder(BertEncoder):
    """
    Custom encoder that, after a chosen layer (global_layer_index), injects a global dense vector.
    
    Args:
        config: BertConfig
        global_layer_index (int): Index of the layer after which to apply the global dense aggregator.
        global_dim (int): The dimension N of the global vector.
        seq_length (int): The fixed sequence length.
    """
    def __init__(self, config, global_layer_index, global_dim, seq_length):
        super().__init__(config)
        self.global_layer_index = global_layer_index
        self.global_dense = GlobalDenseAggregator(seq_length=seq_length,
                                                  hidden_size=config.hidden_size,
                                                  global_dim=global_dim)
        # Create the usual BERT layers
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
            
            # After the chosen layer index, apply the global dense aggregator
            if i == self.global_layer_index:
                hidden_states = self.global_dense(hidden_states)
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            pooler_output=None,
        )

class CustomBertForTokenClassification(BertPreTrainedModel):
    """
    A BertForTokenClassification model with a custom global dense layer inserted in the encoder.
    
    Args:
        config: BertConfig
        global_layer_index (int): The encoder layer after which to inject the global dense layer.
        global_dim (int): The size N of the global vector.
        seq_length (int): The fixed context length (e.g. 512).
    """
    def __init__(self, config, global_layer_index, global_dim, seq_length):
        global_layer_index = config.pop('global_layer_index')
        global_dim = config.pop('global_vector_dim')
        seq_length = config.pop('fixed_seq_length')
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Load the standard BERT model and then replace its encoder with our custom encoder.
        self.bert = BertModel(config)
        self.bert.encoder = CustomBertEncoder(config,
                                              global_layer_index=global_layer_index,
                                              global_dim=global_dim,
                                              seq_length=seq_length)
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights (using the method from BertPreTrainedModel)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        # outputs.last_hidden_state shape: (batch_size, seq_length, hidden_size)
        sequence_output = outputs.last_hidden_state
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten the predictions and labels for token classification.
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return (loss, logits) if loss is not None else logits
