import copy
import os
import threading
from dataclasses import dataclass, field

from modules.util import path_util
from modules.util.enum.ConceptType import ConceptType
from modules.util.image_util import load_image

import torch
import torch.nn.functional as F
from torch import Tensor

import imagesize
from PIL import Image, ImageDraw, ImageFont

SLIDER_IMAGE_AUGMENTATION_FIELDS = (
    "enable_crop_jitter",
    "enable_random_flip",
    "enable_fixed_flip",
    "enable_random_rotate",
    "enable_fixed_rotate",
    "enable_random_brightness",
    "enable_fixed_brightness",
    "enable_random_contrast",
    "enable_fixed_contrast",
    "enable_random_saturation",
    "enable_fixed_saturation",
    "enable_random_hue",
    "enable_fixed_hue",
    "enable_random_circular_mask_shrink",
    "enable_random_mask_rotate_crop",
)

SLIDER_TEXT_AUGMENTATION_FIELDS = (
    "enable_tag_shuffling",
    "tag_dropout_enable",
    "caps_randomize_enable",
)


@dataclass(frozen=True)
class SliderPair:
    key: str
    pos_path: str
    neg_path: str
    width: int | None = None
    height: int | None = None


@dataclass
class SliderPairScanResult:
    matched_pairs: list[SliderPair] = field(default_factory=list)
    unmatched_positive: list[str] = field(default_factory=list)
    unmatched_negative: list[str] = field(default_factory=list)
    mismatched_dimensions: list[tuple[str, str, str, tuple[int, int], tuple[int, int]]] = field(default_factory=list)
    duplicate_positive: list[str] = field(default_factory=list)
    duplicate_negative: list[str] = field(default_factory=list)
    dimensions_summary: dict[tuple[int, int], int] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def found_pairs_count(self) -> int:
        return len(self.matched_pairs) + len(self.mismatched_dimensions)

    @property
    def matching_pairs_count(self) -> int:
        return len(self.matched_pairs)

    @property
    def unmatched_pairs_count(self) -> int:
        return len(self.unmatched_positive) + len(self.unmatched_negative) + len(self.mismatched_dimensions)


def disable_concept_augmentations_dict(concept_dict: dict) -> dict:
    updated = copy.deepcopy(concept_dict)
    image_cfg = updated.get("image")
    if isinstance(image_cfg, dict):
        for field_name in SLIDER_IMAGE_AUGMENTATION_FIELDS:
            if field_name in image_cfg:
                image_cfg[field_name] = False
    text_cfg = updated.get("text")
    if isinstance(text_cfg, dict):
        for field_name in SLIDER_TEXT_AUGMENTATION_FIELDS:
            if field_name in text_cfg:
                text_cfg[field_name] = False
    return updated


def check_mixed_slider_concepts(concepts: list) -> tuple[bool, bool, bool]:
    has_slider = False
    has_non_slider_training = False
    for concept in concepts:
        enabled = concept.get("enabled", True) if isinstance(concept, dict) else getattr(concept, "enabled", True)
        if not enabled:
            continue
        raw_type = concept.get("type", ConceptType.STANDARD) if isinstance(concept, dict) else getattr(concept, "type", ConceptType.STANDARD)
        concept_type = ConceptType(raw_type)
        if concept_type == ConceptType.VALIDATION:
            continue
        if concept_type == ConceptType.SLIDER:
            has_slider = True
        else:
            has_non_slider_training = True
    return has_slider, has_non_slider_training, has_slider and has_non_slider_training


def warn_if_mixed_slider_concepts(concepts: list) -> bool:
    _, _, is_mixed = check_mixed_slider_concepts(concepts)
    if is_mixed:
        print(
            "[Slider] WARNING: Both SLIDER and non-SLIDER training concepts are active! "
            "Combining slider concepts with non-slider concept types in the same training run "
            "can cause unexpected behavior and training issues."
        )
    return is_mixed


def get_image_dimensions(path: str) -> tuple[int, int]:
    try:
        width, height = imagesize.get(path)
        if width > 0 and height > 0:
            return int(width), int(height)
    except Exception:
        pass
    img = load_image(path)
    try:
        return int(img.width), int(img.height)
    finally:
        img.close()


def _collect_image_stems(
        root_dir: str,
        include_subdirectories: bool,
        extensions: set[str],
        exclude_postfix: tuple[str, ...],
        cancel_event: threading.Event | None = None,
) -> tuple[dict[str, str], list[str], bool]:
    files_by_key: dict[str, str] = {}
    duplicates: list[str] = []
    if not root_dir or not os.path.isdir(root_dir):
        return files_by_key, duplicates, False

    candidates: list[tuple[str, str]] = []
    if include_subdirectories:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if cancel_event is not None and cancel_event.is_set():
                return files_by_key, duplicates, True
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")
                candidates.append((rel_path, full_path))
    else:
        for entry in sorted(os.scandir(root_dir), key=lambda e: e.name):
            if cancel_event is not None and cancel_event.is_set():
                return files_by_key, duplicates, True
            if entry.name.startswith(".") or not entry.is_file():
                continue
            candidates.append((entry.name, entry.path))

    for rel_path, full_path in candidates:
        if cancel_event is not None and cancel_event.is_set():
            return files_by_key, duplicates, True
        stem_rel, ext = os.path.splitext(rel_path)
        if ext.lower() not in extensions:
            continue
        if any(stem_rel.endswith(postfix) for postfix in exclude_postfix):
            continue
        if stem_rel in files_by_key:
            duplicates.append(rel_path)
            continue
        files_by_key[stem_rel] = path_util.canonical_join(full_path)

    return files_by_key, duplicates, False


def scan_slider_pairs(
        pos_dir: str,
        neg_dir: str,
        include_subdirectories: bool = False,
        extensions: set[str] | None = None,
        exclude_postfix: tuple[str, ...] = ("-masklabel", "-condlabel"),
        check_dimensions: bool = True,
        cancel_event: threading.Event | None = None,
) -> SliderPairScanResult:
    if extensions is None:
        extensions = {ext.lower() for ext in path_util.supported_image_extensions()}
    else:
        extensions = {ext.lower() for ext in extensions}

    result = SliderPairScanResult()
    pos_map, dup_pos, cancelled = _collect_image_stems(
        pos_dir, include_subdirectories, extensions, exclude_postfix, cancel_event
    )
    if cancelled:
        result.cancelled = True
        return result
    neg_map, dup_neg, cancelled = _collect_image_stems(
        neg_dir, include_subdirectories, extensions, exclude_postfix, cancel_event
    )
    if cancelled:
        result.cancelled = True
        return result

    result.duplicate_positive = dup_pos
    result.duplicate_negative = dup_neg

    all_keys = sorted(set(pos_map.keys()) | set(neg_map.keys()))
    for key in all_keys:
        if cancel_event is not None and cancel_event.is_set():
            result.cancelled = True
            return result
        in_pos = key in pos_map
        in_neg = key in neg_map
        if in_pos and not in_neg:
            result.unmatched_positive.append(os.path.relpath(pos_map[key], pos_dir).replace("\\", "/"))
            continue
        if in_neg and not in_pos:
            result.unmatched_negative.append(os.path.relpath(neg_map[key], neg_dir).replace("\\", "/"))
            continue

        pos_path = pos_map[key]
        neg_path = neg_map[key]
        if check_dimensions:
            try:
                pos_size = get_image_dimensions(pos_path)
                neg_size = get_image_dimensions(neg_path)
            except Exception:
                # Let image loader handle/report corrupt files if unreadable; skip dimension mismatch check here.
                result.matched_pairs.append(SliderPair(key=key, pos_path=pos_path, neg_path=neg_path))
                continue

            if pos_size != neg_size:
                result.mismatched_dimensions.append((key, pos_path, neg_path, pos_size, neg_size))
                continue
            w, h = pos_size
            result.dimensions_summary[(w, h)] = result.dimensions_summary.get((w, h), 0) + 1
            result.matched_pairs.append(SliderPair(key=key, pos_path=pos_path, neg_path=neg_path, width=w, height=h))
        else:
            result.matched_pairs.append(SliderPair(key=key, pos_path=pos_path, neg_path=neg_path))

    return result


def log_slider_scan_result(concept_name: str, result: SliderPairScanResult) -> None:
    label = concept_name or "(unnamed)"
    print(
        f"[Slider] Concept '{label}': found {result.matching_pairs_count} matching slider pair(s) "
        f"(by name: {result.found_pairs_count}, unmatched +1: {len(result.unmatched_positive)}, "
        f"unmatched -1: {len(result.unmatched_negative)}, dimension mismatches: {len(result.mismatched_dimensions)})."
    )

    if result.dimensions_summary:
        dims_str = ", ".join(
            f"{w}x{h} ({count})"
            for (w, h), count in sorted(result.dimensions_summary.items())
        )
        print(f"[Slider] Concept '{label}' matching pair dimensions: {dims_str}")

    for dup in result.duplicate_positive:
        print(f"[Slider] WARNING: Duplicate stem in +1 folder for '{label}', skipping: {dup}")
    for dup in result.duplicate_negative:
        print(f"[Slider] WARNING: Duplicate stem in -1 folder for '{label}', skipping: {dup}")
    for rel in result.unmatched_positive:
        print(f"[Slider] WARNING: Unmatched +1 image (no pair in -1 folder), skipping: {rel}")
    for rel in result.unmatched_negative:
        print(f"[Slider] WARNING: Unmatched -1 image (no pair in +1 folder), skipping: {rel}")
    for key, _pos_p, _neg_p, (pw, ph), (nw, nh) in result.mismatched_dimensions:
        print(
            f"[Slider] WARNING: Dimension mismatch for pair '{key}' "
            f"(+1: {pw}x{ph} vs -1: {nw}x{nh}), skipping."
        )


def _normalize_diff_map(diff: Tensor, diff_weight: float) -> Tensor:
    spatial_dims = tuple(range(1, diff.ndim))
    dm = diff.mean(dim=spatial_dims, keepdim=True)
    r = (diff / dm.clamp_min(1e-8)).clamp(max=8.0)
    w = (1.0 - float(diff_weight)) + float(diff_weight) * r
    w = w / w.mean(dim=spatial_dims, keepdim=True).clamp_min(1e-8)
    return torch.where(dm > 1e-6, w, torch.ones_like(w))


@torch.no_grad()
def compute_slider_diff_weight(
        model,
        pos_latent: Tensor,
        neg_latent: Tensor,
        diff_weight: float,
) -> Tensor | None:
    if diff_weight <= 0.0:
        return None

    model_type = getattr(model, "model_type", None)
    has_vae = getattr(model, "vae", None) is not None
    pos_f = pos_latent.float()
    neg_f = neg_latent.float()

    if has_vae and model_type is not None and model_type.is_moss():
        pos_s = model.scale_latents(pos_f)
        neg_s = model.scale_latents(neg_f)
        d = (pos_s - neg_s).abs().mean(dim=1, keepdim=True)
        return _normalize_diff_map(d, diff_weight)

    if has_vae and model_type is not None and (model_type.is_flux_2() or model_type.is_ernie()):
        pos_s = model.scale_latents(model.patchify_latents(pos_f))
        neg_s = model.scale_latents(model.patchify_latents(neg_f))
        d = (pos_s - neg_s).abs().mean(dim=1, keepdim=True)
        w = _normalize_diff_map(d, diff_weight)
        return w.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    if has_vae and model_type is not None and model_type.is_ideogram():
        grid_h = pos_f.shape[-2] // 2
        grid_w = pos_f.shape[-1] // 2
        pos_s = model.scale_latents(model.patchify_latents(pos_f))
        neg_s = model.scale_latents(model.patchify_latents(neg_f))
        d = (pos_s - neg_s).abs().mean(dim=-1)
        d = d.reshape(pos_f.shape[0], 1, grid_h, grid_w)
        w = _normalize_diff_map(d, diff_weight)
        return w.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    if has_vae and model_type is not None and (model_type.is_krea2() or model_type.is_qwen()):
        pos_s = model.scale_latents(pos_f)
        neg_s = model.scale_latents(neg_f)
        d = (pos_s - neg_s).abs().mean(dim=1, keepdim=True)
        if d.ndim == 5 and d.shape[-2] % 2 == 0 and d.shape[-1] % 2 == 0:
            b, c, f, h, w_dim = d.shape
            d_2d = F.avg_pool2d(d.reshape(b * f, c, h, w_dim), kernel_size=2, stride=2)
            d_pooled = d_2d.reshape(b, c, f, h // 2, w_dim // 2)
            w = _normalize_diff_map(d_pooled, diff_weight)
            return w.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        return _normalize_diff_map(d, diff_weight)

    if has_vae and model_type is not None and model_type.is_anima():
        pos_s = model.scale_latents(pos_f)
        neg_s = model.scale_latents(neg_f)
        d = (pos_s - neg_s).abs().mean(dim=1, keepdim=True)
        return _normalize_diff_map(d, diff_weight)

    d = (pos_f - neg_f).abs().mean(dim=1, keepdim=True)
    return _normalize_diff_map(d, diff_weight)


def stitch_slider_images(
        images: list[Image.Image],
        multipliers: tuple[float, ...] = (-1.0, 1.0),
) -> Image.Image:
    if len(images) != len(multipliers):
        raise ValueError(f"Expected {len(multipliers)} images to stitch, got {len(images)}")
    if not images:
        raise ValueError("Cannot stitch an empty list of images")

    rgb_images = [img.convert("RGB") for img in images]
    width, height = rgb_images[0].size
    gap = 8
    try:
        font = ImageFont.load_default(size=max(14, width // 40))
    except TypeError:
        font = ImageFont.load_default()

    labels = ["Slider 0" if m == 0 else f"Slider {m:+g}" for m in multipliers]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bboxes = [probe.textbbox((0, 0), label, font=font) for label in labels]
    top = min(box[1] for box in bboxes)
    bottom = max(box[3] for box in bboxes)
    pad = max(6, (bottom - top) // 3)
    band = (bottom - top) + 2 * pad

    strip = Image.new("RGB", (width * len(rgb_images) + gap * (len(rgb_images) - 1), height + band), (16, 16, 16))
    draw = ImageDraw.Draw(strip)
    for idx, (img, label) in enumerate(zip(rgb_images, labels, strict=True)):
        x = idx * (width + gap)
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.BILINEAR)
        strip.paste(img, (x, band))
        draw.text((x + 8, pad - top), label, fill=(236, 236, 236), font=font)
    return strip


def stitch_slider_videos(videos: list[Tensor]) -> Tensor:
    if not videos:
        raise ValueError("Cannot stitch an empty list of videos")
    tensors = [v.detach().cpu() for v in videos]
    if tensors[0].ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {tensors[0].shape}")
    # (T, H, W, C) when last dim is 3 channels; otherwise (C, T, H, W)
    width_dim = 2 if tensors[0].shape[-1] == 3 else 3
    return torch.cat(tensors, dim=width_dim)
