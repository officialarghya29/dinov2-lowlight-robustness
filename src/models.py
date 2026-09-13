"""Backbone loading, preprocessing (incl. float-input arm), and embedding hooks.

Float-input handling matters scientifically: torchvision's ToTensor only
rescales uint8 input (/255). The quantization experiment feeds float stage-1
images, so the /255 must be applied manually there — otherwise the two arms of
the experiment would differ by a 255x scale factor instead of by quantization.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def load_dinov2(model_name: str = "dinov2_vits14", device: str = "cuda") -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


def build_uint8_preprocess(input_size: int = INPUT_SIZE) -> T.Compose:
    """For uint8 HWC arrays. ToTensor handles the /255."""
    return T.Compose([
        T.ToTensor(),
        T.Resize((input_size, input_size), antialias=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def float_to_tensor(img_f32: np.ndarray, input_size: int = INPUT_SIZE) -> torch.Tensor:
    """For float32 HWC arrays in [0, 255]: manual /255 + resize + normalize."""
    t = torch.from_numpy(img_f32.astype(np.float32)).permute(2, 0, 1) / 255.0
    t = T.functional.resize(t, (input_size, input_size), antialias=True)
    return T.functional.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)


class EmbeddingExtractor:
    """CLS-token embedding extraction with optional per-block hooks.

    collect_layers=True captures each transformer block's CLS output for
    layer-wise CKA. Hooks are released after use to avoid leaking tensors.
    """

    def __init__(self, model: torch.nn.Module, device: str, input_size: int = INPUT_SIZE):
        self.model = model
        self.device = device
        self.preprocess = build_uint8_preprocess(input_size)
        self._hooks = []
        self._layer_outputs = {}

    def _install_hooks(self):
        for i, block in enumerate(self.model.blocks):
            def make_hook(idx):
                def hook(module, inp, out):
                    self._layer_outputs[idx] = out[:, 0, :].detach().float().cpu()
                return hook
            self._hooks.append(block.register_forward_hook(make_hook(i)))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._layer_outputs = {}

    @torch.no_grad()
    def __call__(self, image_list, batch_size: int = 64, collect_layers: bool = False,
                 float_input: bool = False):
        if collect_layers:
            self._install_hooks()
        pooled, layerwise = [], {}
        try:
            for i in range(0, len(image_list), batch_size):
                batch = image_list[i:i + batch_size]
                if float_input:
                    tensors = torch.stack([float_to_tensor(img) for img in batch]).to(self.device)
                else:
                    tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
                out = self.model(tensors)
                if out.dim() == 3:  # HF-style output guard
                    out = out[:, 0, :]
                pooled.append(out.float().cpu())
                if collect_layers:
                    for l, v in self._layer_outputs.items():
                        layerwise.setdefault(l, []).append(v)
                    self._layer_outputs.clear()
        finally:
            if collect_layers:
                self._remove_hooks()
        pooled = torch.cat(pooled, dim=0).numpy()
        if collect_layers:
            layerwise = {l: torch.cat(v, dim=0).numpy() for l, v in layerwise.items()}
            return pooled, layerwise
        return pooled
