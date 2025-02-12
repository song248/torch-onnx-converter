import torch.nn as nn
from transformers.models.clip.configuration_clip import (
    CLIPConfig,
)  # , CLIPTextConfig, CLIPVisionConfig
from CLIP_VIP.CLIP_modules import CLIPModel
from CLIP_VIP.base import T2VRetConfig
from types import SimpleNamespace


class CLIPVIP(nn.Module):
    def __init__(self, args: T2VRetConfig, state_dict):
        super(CLIPVIP, self).__init__()
        clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        setattr(
            clipconfig,
            "vision_additional_config",
            SimpleNamespace(
                **{
                    "temporal_size": args.temporal_size,
                    "if_use_temporal_embed": 1,
                    "logit_scale_init_value": state_dict[
                        "clipmodel.logit_scale"
                    ].item(),
                    "add_cls_num": state_dict[
                        "clipmodel.vision_model.embeddings.added_cls"
                    ].shape[0],
                }
            ),
        )
        self.clipmodel = CLIPModel(clipconfig)
        logit_scale_value = state_dict["clipmodel.logit_scale"].item()
        self.clipmodel.logit_scale.data.fill_(logit_scale_value)

    def overload_logit_scale(self, overload_logit_scale):
        self.clipmodel.logit_scale.data.fill_(overload_logit_scale)

    def forward(
        self,
        video,
        text_input_ids,
        text_input_mask,
        image=None,
        caption_ids=None,
        caption_masks=None,
    ):
        """
        video [B, n_clips*num_frms, C, H, W]
        text_input_ids [B, L]
        text_input_mask [B, L]
        image [B, img_num, C, H, W]
        caption_ids [B, img_num, L]
        caption_masks [B, img_num, L]
        """
        _, _, C, H, W = video.shape

        inputs = {
            "input_ids": text_input_ids,
            "attention_mask": text_input_mask,
            "pixel_values": video,
            "return_loss": False,
        }
        outputs = self.clipmodel(**inputs)
        results = {}
        results["text_features"] = outputs["text_embeds"]
        results["vis_features"] = outputs["image_embeds"]

        if image is not None:
            _, _, C, H, W = image.shape
            L = caption_ids.shape[-1]
            inputs = {
                "input_ids": caption_ids.reshape(-1, L),
                "attention_mask": caption_masks.reshape(-1, L),
                "pixel_values": image.reshape(-1, 1, C, H, W),
                "return_loss": False,
            }
            outputs = self.clipmodel(**inputs)
            results["img_features"] = outputs["image_embeds"]
            results["cap_features"] = outputs["text_embeds"]

        return results

    def encode_video(self, video):
        inputs = {"pixel_values": video, "if_norm": True}
        video_features = self.clipmodel.get_image_features(**inputs)
        return video_features

    def encode_text(self, text_input_ids, text_input_mask):
        inputs = {
            "input_ids": text_input_ids,
            "attention_mask": text_input_mask,
            "if_norm": True,
        }
        text_features = self.clipmodel.get_text_features(**inputs)
        return text_features
