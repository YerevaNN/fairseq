from fairseq.models import register_model
from fairseq.models.transformer_lm import TransformerLanguageModel, TransformerLanguageModelConfig
from fairseq.models.transformer.diffusion_transformer_decoder import DiffusionTransformerDecoder


@register_model("diffusion_transformer_lm", dataclass=TransformerLanguageModelConfig)
class DiffusionTransformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    # def forward(self, *args, **kwargs):
    #     out = super().forward(*args, **kwargs)
    #     return out

    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        model.decoder = DiffusionTransformerDecoder(
            args, task.target_dictionary, model.decoder.embed_tokens, no_encoder_attn=True
        )
        return model
