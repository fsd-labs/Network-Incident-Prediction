from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
import logging
logger = logging.getLogger(__name__)

from ..modelling.logbert_model import BERT, ClassifierHead, MaskedLogModel
from ..modelling.interpolate import Interpolate
from .config import LogBertConfig


class LogBertPreTrainedModel(PreTrainedModel):
    config_class = LogBertConfig
    base_model_prefix = "logbert"

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


class LogBertModel(LogBertPreTrainedModel):
    def __init__(self, config: LogBertConfig):
        super().__init__(config)
        self.bert = BERT(
            vocab_size=config.vocab_size,
            max_len=config.max_position_embeddings,
            hidden=config.hidden_size,
            n_layers=config.num_hidden_layers,
            attn_heads=config.num_attention_heads,
            dropout=0.1,
            is_logkey=True,
            is_time=config.is_time,
            is_device=config.is_device,
            num_devices=config.num_devices,
            causal=config.causal,
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        device_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        hidden = self.bert(
            x=input_ids,
            device_info=device_ids,
        )
        return BaseModelOutput(last_hidden_state=hidden)


class LogBertForSequenceClassification(LogBertPreTrainedModel):
    def __init__(self, config: LogBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.use_mlm = config.use_mlm
        self.use_l1 = config.use_l1

        # Match original state_dict keys exactly: bert, interpolate, classifier, (mask_lm)
        self.bert = BERT(
            vocab_size=config.vocab_size,
            max_len=config.max_position_embeddings,
            hidden=config.hidden_size,
            n_layers=config.num_hidden_layers,
            attn_heads=config.num_attention_heads,
            dropout=0.1,
            is_logkey=True,
            is_time=config.is_time,
            is_device=config.is_device,
            num_devices=config.num_devices,
            causal=config.causal,
        )
        self.interpolate = Interpolate(max_len_seq=config.max_position_embeddings)
        self.classifier = ClassifierHead(hidden=config.hidden_size, max_len_seq=config.max_position_embeddings, num_classes=config.num_labels)
        if self.use_mlm:
            self.mask_lm = MaskedLogModel(self.hidden_size, config.vocab_size)
        else:
            self.mask_lm = None

        self.class_weight = None  # can be set externally by trainer
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.mlm_loss_fn = nn.NLLLoss(ignore_index=0)
        # Register class_weight as a buffer so it moves with the model during DDP
        self.register_buffer("_class_weight_buffer", None)
        self.post_init()
        # self._init_agg_as_mean()

    def _init_agg_as_mean(self):
        with torch.no_grad():
            n = self.classifier.max_len_seq  
            self.classifier.linear1.weight.fill_(1.0 / n)
            if self.classifier.linear1.bias is not None:
                self.classifier.linear1.bias.zero_()

    def set_class_weight(self, weight: Optional[torch.Tensor]):
        """Set class weight for imbalanced classification.
        
        Note: The weight is registered as a buffer and the loss_fn is recreated
        in forward() to ensure correct device placement in DDP training.
        """
        if weight is not None:
            # Register as buffer so it moves with the model
            self.register_buffer("_class_weight_buffer", weight.clone())
            self.class_weight = weight
        else:
            self._class_weight_buffer = None
            self.class_weight = None

    def __cal_l1_norm(self):
        w_agr = self.classifier.linear1.weight
        l1 = torch.norm(w_agr, 1)
        return l1


    def forward(
        self,
        input_ids: torch.LongTensor,
        device_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        
        x = self.bert(x=input_ids, device_info=device_ids)  # [B, L, H]

        attn_mask = (input_ids > 0).float()
        attn_mask[:, 0] = 0

        x_norm = self.interpolate.normalize_sequence(x, attn_mask=attn_mask)
        logits = self.classifier(x_norm)

        loss = None
        loss_cls_value = None
        loss_mlm_value = None
        l1_loss_value = None
        if labels is not None:
            # Apply class weights only during training; eval should report unweighted CE.
            if self.training and self._class_weight_buffer is not None:
                loss_cls_value = nn.functional.cross_entropy(
                    logits, labels, weight=self._class_weight_buffer, reduction="mean"
                )
            else:
                loss_cls_value = self.loss_fn(logits, labels)
            loss = loss_cls_value
        if self.use_l1:
            l1_loss_value = self.__cal_l1_norm()
            if loss is None:
                loss = l1_loss_value
            else:
                # lambda = 
                loss = loss + 1e-4 * l1_loss_value

        if self.use_mlm and mlm_labels is not None and self.mask_lm is not None:
            mlm_out = self.mask_lm(x)
            ignore_index = self.mlm_loss_fn.ignore_index  # = 0
            valid_mask = (mlm_labels != ignore_index)
            # invalid_mask = ~valid_mask
            # if valid_mask.any():
            #     logger.warning(f"NONERROR LABELS: {mlm_labels[valid_mask].detach().cpu()}")
            #     logger.warning(f"INPUT: {input_ids[valid_mask].detach().cpu()}")

            if valid_mask.any():
                loss_mlm_value = self.mlm_loss_fn(mlm_out.transpose(1, 2), mlm_labels)
            else:
                loss_mlm_value = x.new_zeros(())

            if loss is None:
                loss = loss_mlm_value
            else:
                loss = loss + 0.1 * loss_mlm_value

        
        out = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
        out.loss_cls = loss_cls_value
        out.loss_mlm = loss_mlm_value
        out.loss_l1 = l1_loss_value
        # Ensure these fields are visible in dict-like access for Trainer hooks
        try:
            out["loss_cls"] = loss_cls_value
            out["loss_mlm"] = loss_mlm_value
            out["loss_l1"] = l1_loss_value
        except Exception:
            pass
        return out
