"""
参考图生成（底模默认 DreamShaper / Lykon/dreamshaper-8，见 reference_modes.json 的 sd15_base）：
按模式懒加载 ControlNet / IP-Adapter，互斥占用显存。
预处理依赖 controlnet_aux（OpenPose/HED/Lineart）；Canny 为 OpenCV。
"""
from __future__ import annotations

import gc
import json
import os
from typing import Any

import torch
from PIL import Image

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_CONFIG: dict[str, Any] | None = None
_preprocessors: dict[str, Any] = {}
_reference_bundle: dict[str, Any] | None = None


def load_reference_config() -> dict[str, Any]:
    global _REFERENCE_CONFIG
    if _REFERENCE_CONFIG is None:
        path = os.path.join(_PROJECT_DIR, "reference_modes.json")
        with open(path, encoding="utf-8") as f:
            _REFERENCE_CONFIG = json.load(f)
    return _REFERENCE_CONFIG


def get_mode_config(mode: str) -> dict[str, Any]:
    cfg = load_reference_config()
    modes = cfg.get("modes") or {}
    if mode not in modes:
        raise KeyError(mode)
    return modes[mode]


def unload_reference_bundle() -> None:
    global _reference_bundle, _preprocessors
    if _reference_bundle is not None:
        pipe = _reference_bundle.get("pipeline")
        if pipe is not None:
            try:
                if hasattr(pipe, "unload_ip_adapter"):
                    pipe.unload_ip_adapter()
            except Exception:
                pass
            del pipe
        _reference_bundle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _hf_local_first(load, desc: str, local_files_only: bool, allow_hub: bool):
    try:
        return load(True)
    except Exception as e:
        if local_files_only or not allow_hub:
            raise RuntimeError(
                f"{desc}: not in Hugging Face cache. Pre-download or set ALLOW_HF_HUB_DOWNLOAD=true."
            ) from e
        return load(False)


def load_reference_pipeline(
    mode: str,
    *,
    hf_token: str | None,
    local_files_only: bool,
    allow_hub: bool,
) -> dict[str, Any]:
    """加载当前模式所需权重；若模式未变则复用。"""
    global _reference_bundle
    mc = get_mode_config(mode)
    mtype = mc["type"]

    if _reference_bundle is not None and _reference_bundle.get("mode") == mode:
        return _reference_bundle

    unload_reference_bundle()

    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        StableDiffusionPipeline,
    )

    cfg = load_reference_config()
    base = cfg["sd15_base"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if mtype == "controlnet":
        cn_repo = mc["controlnet_repo"]

        def load_cn(lfo: bool):
            return ControlNetModel.from_pretrained(
                cn_repo,
                torch_dtype=dtype,
                token=hf_token,
                local_files_only=lfo,
            )

        cn = _hf_local_first(
            load_cn,
            f"ControlNet ({cn_repo})",
            local_files_only,
            allow_hub,
        )

        def load_pipe(lfo: bool):
            return StableDiffusionControlNetPipeline.from_pretrained(
                base,
                controlnet=cn,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                token=hf_token,
                local_files_only=lfo,
            )

        pipe = _hf_local_first(
            load_pipe,
            f"SD1.5+ControlNet ({base})",
            local_files_only,
            allow_hub,
        )
        pipe = pipe.to(device)
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        _reference_bundle = {"mode": mode, "type": mtype, "pipeline": pipe, "device": device}
        return _reference_bundle

    if mtype == "ip_adapter":

        def load_txt(lfo: bool):
            return StableDiffusionPipeline.from_pretrained(
                base,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                token=hf_token,
                local_files_only=lfo,
            )

        pipe = _hf_local_first(load_txt, f"SD1.5 ({base})", local_files_only, allow_hub)
        pipe = pipe.to(device)
        repo = mc["ip_adapter_repo"]
        sub = mc.get("ip_adapter_subfolder", "models")
        wname = mc["ip_adapter_weight"]
        try:
            pipe.load_ip_adapter(
                repo,
                subfolder=sub,
                weight_name=wname,
                token=hf_token,
            )
        except TypeError:
            pipe.load_ip_adapter(repo, subfolder=sub, weight_name=wname)
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        _reference_bundle = {"mode": mode, "type": mtype, "pipeline": pipe, "device": device}
        return _reference_bundle

    raise ValueError(f"Unknown reference mode type: {mtype}")


def _preprocess_canny_opencv(image: Image.Image, size: int) -> Image.Image:
    """与旧版 /process-canny 一致：灰度 + Canny(100,200)，三通道；再缩放到 size。"""
    import cv2
    import numpy as np

    img_array = np.array(image.convert("RGB"))
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    low_threshold, high_threshold = 100, 200
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    out = Image.fromarray(edges).convert("RGB")
    if out.size != (size, size):
        out = out.resize((size, size), Image.LANCZOS)
    return out


def _get_preprocessor(name: str):
    """懒加载 controlnet_aux 检测器（权重来自 lllyasviel/Annotators）。"""
    global _preprocessors
    if name in _preprocessors:
        return _preprocessors[name]
    try:
        from controlnet_aux import HEDdetector, OpenposeDetector  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "需要安装 controlnet-aux 以使用 OpenPose/HED/Lineart 预处理：pip install controlnet-aux"
        ) from e

    annot = "lllyasviel/Annotators"
    if name == "openpose":
        _preprocessors[name] = OpenposeDetector.from_pretrained(annot)
    elif name == "hed":
        _preprocessors[name] = HEDdetector.from_pretrained(annot)
    elif name == "lineart":
        try:
            from controlnet_aux import LineartDetector  # type: ignore

            _preprocessors[name] = LineartDetector.from_pretrained(annot)
        except Exception:
            from controlnet_aux.lineart import LineartStandardDetector  # type: ignore

            _preprocessors[name] = LineartStandardDetector.from_pretrained(annot)
    else:
        raise ValueError(f"Unknown preprocessor: {name}")
    return _preprocessors[name]


def preprocess_reference_image(mode: str, image: Image.Image) -> Image.Image:
    """将用户 RGB 图转为当前模式所需的控制图（或原图），并缩放到 output_size。"""
    size = get_output_size()
    mc = get_mode_config(mode)
    mtype = mc["type"]
    if mtype == "ip_adapter":
        out = image.convert("RGB")
        if out.size != (size, size):
            out = out.resize((size, size), Image.LANCZOS)
        return out

    prep = mc.get("preprocessor")
    if not prep:
        out = image.convert("RGB")
        if out.size != (size, size):
            out = out.resize((size, size), Image.LANCZOS)
        return out
    if prep == "canny":
        return _preprocess_canny_opencv(image, size)
    det = _get_preprocessor(prep)
    arr = det(image)
    if isinstance(arr, Image.Image):
        out = arr.convert("RGB")
    else:
        import numpy as np

        a = np.asarray(arr)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        out = Image.fromarray(a).convert("RGB")
    if out.size != (size, size):
        out = out.resize((size, size), Image.LANCZOS)
    return out


def get_output_size() -> int:
    cfg = load_reference_config()
    return int(cfg.get("output_size", 768))


def get_loaded_reference_mode() -> str | None:
    if _reference_bundle is None:
        return None
    return str(_reference_bundle.get("mode", ""))
