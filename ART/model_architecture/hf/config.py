from transformers import PretrainedConfig

class LogBertConfig(PretrainedConfig):
    model_type = "logbert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=20480,
        is_time=False,
        is_device=True,
        num_devices=2,
        use_mlm=True,
        use_l1=True,
        num_labels=2,
        causal=False,
        pad_token_id=0,
        
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.is_time = is_time
        self.is_device = is_device
        self.num_devices = num_devices
        self.use_mlm = use_mlm
        self.use_l1 = use_l1
        self.num_labels = num_labels
        self.causal = causal
        # Help downstream loaders (e.g., vLLM/transformers Auto classes)
        self.architectures = ["LogBertForSequenceClassification"]
