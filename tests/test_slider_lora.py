import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch


def _ensure_qt_headless_stub() -> None:
    try:
        from PySide6 import QtGui  # noqa: F401
    except ImportError:
        class _DummyMeta(type):
            def __getattr__(cls, name: str):
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                return cls

        class _DummyBase(metaclass=_DummyMeta):
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return args[0] if len(args) == 1 and callable(args[0]) else _DummyBase()

            def __getattr__(self, name: str):
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                return _DummyBase

        import PySide6  # noqa: PLC0415

        for mod_name in (
            "PySide6.QtGui",
            "PySide6.QtWidgets",
            "PySide6.QtCore",
            "PySide6.QtSvg",
            "matplotlib.backends.backend_qtagg",
        ):
            if mod_name not in sys.modules:
                stub = types.ModuleType(mod_name)
                stub.__file__ = None

                def _getattr(name: str):
                    if name.startswith("__") and name.endswith("__"):
                        raise AttributeError(name)
                    return _DummyBase

                stub.__getattr__ = _getattr  # type: ignore[method-assign]
                sys.modules[mod_name] = stub
                if mod_name.startswith("PySide6."):
                    setattr(PySide6, mod_name.split(".")[1], stub)


_ensure_qt_headless_stub()

import modules.dataLoader.LoadImage as load_image_module
from modules.dataLoader.BaseDataLoader import _slider_safe_collate
from modules.dataLoader.LoadImage import LoadImage
from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.SliderDataModules import (
    CollectSliderPaths,
    OptionalEncodeVAE,
    OptionalImageToVideo,
    OptionalLoadImage,
    OptionalLoadVideo,
    OptionalNormalizeImageChannels,
    OptionalRescaleImageChannels,
    OptionalSampleVAEDistribution,
    OptionalScaleImage,
)
from modules.model.BaseModel import BaseModel
from modules.model.moss.pipeline_moss_image import (
    MOSSImageTransformer,
    _convert_moss_image_transformer_checkpoint,
)
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelSampler.AnimaSampler import AnimaSampler
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSampler.Flux2Sampler import Flux2Sampler
from modules.modelSampler.Krea2Sampler import Krea2Sampler
from modules.modelSampler.MossSampler import MossSampler
from modules.modelSampler.QwenSampler import QwenSampler
from modules.modelSampler.ZImageSampler import ZImageSampler
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.module.FusedModule import FusedModuleGroup
from modules.module.LoRAModule import (
    DoRAModule,
    LoHaModule,
    LoKrModule,
    LoRAModule,
    LoRAModuleWrapper,
    OFTModule,
)
from modules.trainer.GenericTrainer import GenericTrainer
from modules.ui.ConceptTabView import BaseConceptTabView
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.ui.ModelTabController import ModelTabController
from modules.ui.TrainUIController import TrainUIController
from modules.util.concept_stats import (
    folder_scan,
    init_concept_stats,
    update_slider_stats,
    validate_slider_concept,
)
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.slider_util import (
    compute_slider_diff_weight,
    disable_concept_augmentations_dict,
    scan_slider_pairs,
    stitch_slider_images,
    stitch_slider_videos,
)
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch
from torch import nn

import numpy as np
import pytest
from PIL import Image
from safetensors.torch import save_file


def _save_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class _DummyConfigDataset:
    def __init__(self, items: list[dict]):
        self.items = items

    def length(self) -> int:
        return len(self.items)


class _DictListModule(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, key: str, items: list):
        super().__init__()
        self.key = key
        self.items = items

    def length(self) -> int:
        return len(self.items)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return [self.key]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {self.key: self.items[index]}


def test_config_roundtrip_and_backward_compatibility():
    assert str(ConceptType.SLIDER) == "SLIDER"
    assert ConceptType("SLIDER") == ConceptType.SLIDER

    # Legacy JSON without negative_path or slider_diff_weight or slider_multiplier
    concept = ConceptConfig.default_values().from_dict({"name": "legacy", "path": "/tmp/a", "type": "STANDARD"})
    assert concept.negative_path == ""
    assert concept.type == ConceptType.STANDARD

    slider_concept = ConceptConfig.default_values().from_dict({
        "name": "age",
        "path": "/tmp/pos",
        "negative_path": "/tmp/neg",
        "type": "SLIDER",
    })
    assert slider_concept.type == ConceptType.SLIDER
    assert slider_concept.negative_path == "/tmp/neg"
    assert slider_concept.to_dict()["negative_path"] == "/tmp/neg"

    legacy_train_dict = TrainConfig.default_values().to_dict()
    legacy_train_dict.pop("slider_diff_weight", None)
    legacy_train_dict["training_method"] = "LORA"
    train_cfg = TrainConfig.default_values().from_dict(legacy_train_dict)
    assert train_cfg.slider_diff_weight == 1.0
    assert not train_cfg.has_slider_concepts()

    train_cfg.concepts = [slider_concept]
    assert train_cfg.has_slider_concepts()
    slider_concept.enabled = False
    assert not train_cfg.has_slider_concepts()

    sample_cfg = SampleConfig.default_values()
    assert sample_cfg.slider_multiplier == 1.0


def test_pair_scanning_stats_and_augmentation_disabling(tmp_path: Path):
    pos = tmp_path / "pos"
    neg = tmp_path / "neg"
    _save_png(pos / "001.png", (64, 64), (200, 10, 10))
    _save_png(neg / "001.jpg", (64, 64), (10, 10, 200))
    _save_png(pos / "sub" / "002.png", (96, 64), (100, 100, 100))
    _save_png(neg / "sub" / "002.png", (96, 64), (50, 50, 50))
    _save_png(pos / "only_pos.png", (64, 64), (0, 255, 0))
    _save_png(neg / "only_neg.png", (64, 64), (0, 0, 255))
    _save_png(pos / "bad_dim.png", (64, 64), (255, 255, 255))
    _save_png(neg / "bad_dim.png", (128, 64), (0, 0, 0))
    _save_png(pos / "001-masklabel.png", (64, 64), (255, 255, 255))

    scan = scan_slider_pairs(
        str(pos),
        str(neg),
        include_subdirectories=True,
        exclude_postfix=("-masklabel", "-condlabel"),
        check_dimensions=True,
    )
    assert scan.found_pairs_count == 3
    assert scan.matching_pairs_count == 2
    assert scan.unmatched_positive == ["only_pos.png"]
    assert scan.unmatched_negative == ["only_neg.png"]
    assert len(scan.mismatched_dimensions) == 1
    assert scan.dimensions_summary == {(64, 64): 1, (96, 64): 1}

    concept = ConceptConfig.default_values()
    concept.type = ConceptType.SLIDER
    concept.path = str(pos)
    concept.negative_path = str(neg)
    concept.include_subdirectories = True
    concept.image.enable_random_flip = True
    concept.image.enable_resolution_override = True
    concept.image.resolution_override = "512"
    concept.text.tag_dropout_enable = True

    import threading
    import time

    validation = validate_slider_concept(str(pos), str(neg), include_subdirectories=True, concept_name="test_slider")
    assert validation["matching_pairs"] == 2
    assert validation["found_pairs"] == 3
    assert validation["valid"] is False

    stats = init_concept_stats(False)
    stats = folder_scan(
        str(pos),
        stats,
        False,
        concept,
        time.perf_counter(),
        10.0,
        threading.Event(),
    )
    update_slider_stats(stats, str(pos), str(neg), True)
    assert stats["slider_found_pairs"] == 3
    assert stats["slider_matching_pairs"] == 2
    assert stats["slider_unmatched_pairs"] == 2
    assert stats["slider_dimension_mismatches"] == 1
    assert stats["image_count"] == 3

    cleaned = disable_concept_augmentations_dict(concept.to_dict())
    assert cleaned["image"]["enable_random_flip"] is False
    assert cleaned["text"]["tag_dropout_enable"] is False
    # Resolution override must remain preserved
    assert cleaned["image"]["enable_resolution_override"] is True
    assert cleaned["image"]["resolution_override"] == "512"


def test_adapter_multipliers_across_lora_dora_loha_lokr_oft_and_fused():
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    base_linear = nn.Linear(8, 8, bias=False)
    y_base = base_linear(x).detach()

    # Standard LoRA
    lora = LoRAModule("lora_linear", base_linear, rank=2, alpha=2.0)
    lora.hook_to_module()
    with torch.no_grad():
        lora.lora_down.weight.normal_(std=0.2)
        lora.lora_up.weight.normal_(std=0.2)
    lora.set_multiplier(1.0)
    y_pos = lora(x)
    lora.set_multiplier(-1.0)
    y_neg = lora(x)
    lora.set_multiplier(0.0)
    y_zero = lora(x)

    assert torch.allclose(y_zero, y_base, atol=1e-6)
    assert torch.allclose(y_pos - y_base, -(y_neg - y_base), atol=1e-5)
    assert not torch.allclose(y_pos, y_base)
    assert "multiplier" not in lora.state_dict()
    lora.remove_hook_from_module()

    # DoRA
    dora_base = nn.Linear(8, 8, bias=False)
    dora = DoRAModule(
        "lora_dora",
        dora_base,
        2,
        2.0,
        norm_epsilon=False,
        decompose_output_axis=False,
        train_device=torch.device("cpu"),
    )
    dora.hook_to_module()
    with torch.no_grad():
        dora.lora_down.weight.normal_(std=0.2)
        dora.lora_up.weight.normal_(std=0.2)
    dora.set_multiplier(1.0)
    d_pos = dora(x)
    dora.set_multiplier(-1.0)
    d_neg = dora(x)
    assert not torch.allclose(d_pos, d_neg)
    dora.remove_hook_from_module()

    # LoHa
    loha_base = nn.Linear(8, 8, bias=False)
    loha = LoHaModule("mod_loha", loha_base, rank=2, alpha=2.0)
    loha.hook_to_module()
    with torch.no_grad():
        loha.hada_w1_a.normal_(std=0.2)
        loha.hada_w1_b.normal_(std=0.2)
        loha.hada_w2_a.normal_(std=0.2)
        loha.hada_w2_b.normal_(std=0.2)
    loha.set_multiplier(1.0)
    p_out = loha(x)
    loha.set_multiplier(-1.0)
    n_out = loha(x)
    loha.set_multiplier(0.0)
    z_out = loha(x)
    assert torch.allclose(z_out, loha.orig_forward(x), atol=1e-5)
    assert torch.allclose(p_out - z_out, -(n_out - z_out), atol=1e-5)
    loha.remove_hook_from_module()

    # LoKr
    lokr_base = nn.Linear(8, 8, bias=False)
    lokr = LoKrModule(
        "mod_lokr",
        lokr_base,
        dim=2,
        alpha=2.0,
        decompose_both=False,
        decompose_factor=-1,
        use_tucker=False,
        weight_decompose=False,
        dora_on_output=False,
        full_matrix=False,
        train_device=torch.device("cpu"),
    )
    lokr.hook_to_module()
    for p in lokr.parameters():
        with torch.no_grad():
            p.normal_(std=0.2)
    lokr.set_multiplier(1.0)
    lk_pos = lokr(x)
    lokr.set_multiplier(-1.0)
    lk_neg = lokr(x)
    lokr.set_multiplier(0.0)
    lk_zero = lokr(x)
    assert torch.allclose(lk_zero, lokr.orig_forward(x), atol=1e-5)
    assert torch.allclose(lk_pos - lk_zero, -(lk_neg - lk_zero), atol=1e-5)
    lokr.remove_hook_from_module()

    # OFT
    oft_base = nn.Linear(8, 8, bias=False)
    oft = OFTModule("oft_linear", oft_base, oft_block_size=4, block_share=False, oft_scaled=False)
    oft.hook_to_module()
    with torch.no_grad():
        oft.oft_R.weight.normal_(std=0.2)
    oft.set_multiplier(1.0)
    o_pos = oft(x)
    oft.set_multiplier(-1.0)
    o_neg = oft(x)
    oft.set_multiplier(0.0)
    o_zero = oft(x)
    assert torch.allclose(o_zero, oft.orig_forward(x), atol=1e-5)
    assert not torch.allclose(o_pos, o_neg)
    oft.remove_hook_from_module()

    # FusedModuleGroup + LoRAModuleWrapper + BaseModel propagation
    q_leaf = nn.Linear(8, 4, bias=False)
    k_leaf = nn.Linear(8, 4, bias=False)
    fused_group = FusedModuleGroup(
        prefix="fused_qkv",
        leaves=[q_leaf, k_leaf],
        klass=LoRAModule,
        additional_args=[2, 2.0],
        additional_kwargs={},
    )
    fused_group.hook_to_module()
    with torch.no_grad():
        fused_group.module.lora_down.weight.normal_(std=0.2)
        fused_group.module.lora_up.weight.normal_(std=0.2)

    wrapper = LoRAModuleWrapper.__new__(LoRAModuleWrapper)
    wrapper.lora_modules = {"lora_linear": lora, "fused_qkv": fused_group}
    wrapper.fused_groups = [("fused_qkv", ["q", "k"], [q_leaf, k_leaf])]

    class _ToyModel(BaseModel):
        def __init__(self):
            super().__init__(ModelType.FLUX_DEV_1)
            self.transformer_lora = wrapper

    toy = _ToyModel()
    toy.set_adapter_multiplier(-1.0)
    assert float(lora.multiplier.item()) == -1.0
    assert float(fused_group.module.multiplier.item()) == -1.0
    q_neg = q_leaf(x)
    toy.set_adapter_multiplier(1.0)
    assert float(lora.multiplier.item()) == 1.0
    assert float(fused_group.module.multiplier.item()) == 1.0
    q_pos = q_leaf(x)
    assert not torch.allclose(q_pos, q_neg)


def test_slider_diff_weight_architectures_and_loss_application():
    class _FakeModel:
        def __init__(self, model_type: ModelType):
            self.model_type = model_type
            self.vae = object()

        def scale_latents(self, x):
            if self.model_type.is_moss():
                b, c, h, w = x.shape
                return x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
            return x * 0.5

        def patchify_latents(self, x):
            if self.model_type.is_ideogram():
                b, c, h, w = x.shape
                return x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)
            b, c, h, w = x.shape
            return x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)

    # Zero diff weight returns None
    pos4 = torch.zeros(1, 16, 8, 8)
    neg4 = torch.ones(1, 16, 8, 8)
    assert compute_slider_diff_weight(_FakeModel(ModelType.FLUX_DEV_1), pos4, neg4, 0.0) is None

    # Identical latents fall back to all ones
    w_same = compute_slider_diff_weight(_FakeModel(ModelType.FLUX_DEV_1), pos4, pos4, 1.0)
    assert w_same.shape == (1, 1, 8, 8)
    assert torch.allclose(w_same, torch.ones_like(w_same))

    # Localized diff receives higher weight than unchanged region
    pos_loc = torch.zeros(1, 32, 8, 8)
    neg_loc = torch.zeros(1, 32, 8, 8)
    pos_loc[:, :, :4, :4] = 2.0
    for mt, exp_shape in (
        (ModelType.FLUX_DEV_1, (1, 1, 8, 8)),
        (ModelType.FLUX_2, (1, 1, 8, 8)),
        (ModelType.ERNIE, (1, 1, 8, 8)),
        (ModelType.IDEOGRAM_4, (1, 1, 8, 8)),
        (ModelType.MOSS, (1, 1, 4, 4)),
    ):
        w = compute_slider_diff_weight(_FakeModel(mt), pos_loc, neg_loc, 1.0)
        assert w.shape == exp_shape
        assert pytest.approx(float(w.mean().item()), abs=1e-4) == 1.0
        half_h = exp_shape[-2] // 2
        assert w[:, :, :half_h, :half_h].mean() > w[:, :, half_h:, half_h:].mean()

    # 5D Krea2 / Qwen / Anima
    pos5 = torch.zeros(1, 16, 1, 8, 8)
    neg5 = torch.zeros(1, 16, 1, 8, 8)
    pos5[:, :, :, :4, :4] = 2.0
    for mt in (ModelType.KREA_2, ModelType.QWEN, ModelType.ANIMA):
        w5 = compute_slider_diff_weight(_FakeModel(mt), pos5, neg5, 0.8)
        assert w5.shape == (1, 1, 1, 8, 8)
        assert pytest.approx(float(w5.mean().item()), abs=1e-4) == 1.0
        assert w5[:, :, :, :4, :4].mean() > w5[:, :, :, 4:, 4:].mean()

    # Loss mixin verification
    class _LossHost(ModelSetupDiffusionLossMixin):
        pass

    loss_host = _LossHost()
    cfg = TrainConfig.default_values()
    cfg.mse_strength = 1.0
    cfg.mae_strength = 0.5
    cfg.log_cosh_strength = 0.25
    cfg.huber_strength = 0.25
    cfg.loss_weight_fn = LossWeight.CONSTANT
    cfg.loss_scaler = LossScaler.NONE

    pred = torch.zeros(1, 4, 8, 8)
    target = torch.zeros(1, 4, 8, 8)
    target[:, :, :4, :4] = 1.0
    w_high_on_err = torch.zeros(1, 1, 8, 8)
    w_high_on_err[:, :, :4, :4] = 4.0
    w_high_on_zero = torch.zeros(1, 1, 8, 8)
    w_high_on_zero[:, :, 4:, 4:] = 4.0

    sigmas = torch.linspace(0.0, 1.0, 1000)
    data = {"loss_type": "target", "predicted": pred, "target": target, "timestep": torch.tensor([500])}
    loss_a = loss_host._flow_matching_losses({"loss_weight": torch.ones(1), "slider_weight": w_high_on_err}, data, cfg, torch.device("cpu"), sigmas)
    loss_b = loss_host._flow_matching_losses({"loss_weight": torch.ones(1), "slider_weight": w_high_on_zero}, data, cfg, torch.device("cpu"), sigmas)
    assert loss_a.item() > loss_b.item() == 0.0

    # Masked 5D loss test
    cfg.masked_training = True
    pred5 = torch.zeros(1, 4, 2, 8, 8)
    target5 = torch.ones(1, 4, 2, 8, 8)
    mask5 = torch.ones(1, 1, 2, 8, 8)
    w5_sample = torch.full((1, 1, 2, 8, 8), 2.0)
    masked_res = loss_host._flow_matching_losses(
        {"loss_weight": torch.ones(1), "latent_mask": mask5, "slider_weight": w5_sample},
        {"loss_type": "target", "predicted": pred5, "target": target5, "timestep": torch.tensor([500])},
        cfg,
        torch.device("cpu"),
        sigmas,
    )
    assert masked_res.item() > 0.0


def test_mgds_slider_pipeline_and_mixed_collation(tmp_path: Path):
    pos_dir = tmp_path / "pos"
    neg_dir = tmp_path / "neg"
    std_dir = tmp_path / "std"
    _save_png(pos_dir / "a.png", (64, 64), (240, 20, 20))
    _save_png(neg_dir / "a.png", (64, 64), (20, 20, 240))
    _save_png(std_dir / "b.png", (64, 64), (20, 240, 20))

    concepts = [
        {
            "name": "slider_c",
            "path": str(pos_dir),
            "negative_path": str(neg_dir),
            "type": "SLIDER",
            "include_subdirectories": False,
            "enabled": True,
        },
        {
            "name": "std_c",
            "path": str(std_dir),
            "negative_path": "",
            "type": "STANDARD",
            "include_subdirectories": False,
            "enabled": True,
        },
    ]

    collect = CollectSliderPaths(
        concept_in_name="concept",
        path_in_name="path",
        neg_path_in_name="negative_path",
        type_in_name="type",
        include_subdirectories_in_name="concept.include_subdirectories",
        enabled_in_name="enabled",
        path_out_name="image_path",
        neg_path_out_name="negative_image_path",
        concept_out_name="concept",
        extensions={".png"},
        include_postfix=None,
        exclude_postfix=None,
    )
    load_pos = OptionalLoadImage("image_path", "image", 0, 1, {".png"}, dtype=torch.float32)
    load_neg = OptionalLoadImage("negative_image_path", "negative_image", 0, 1, {".png"}, dtype=torch.float32)
    load_neg_vid = OptionalLoadVideo("negative_image_path", "target_frames", "negative_image", 0, 1, 24, {".mp4"}, dtype=torch.float32)
    to_vid = OptionalImageToVideo("negative_image", "negative_image")

    class _DummyVAE:
        def encode(self, x):
            return types.SimpleNamespace(latent_dist=torch.distributions.Normal(x * 0.5, torch.ones_like(x) * 0.1))

    class _DummyEffnet:
        def __call__(self, x):
            return x * 0.25

    prep_mods = [
        RescaleImageChannels("image", "image", 0, 1, -1, 1),
        ScaleImage("image", "image", 0.5),
        SampleVAEDistribution("latent_image_distribution", "latent_image", "mean"),
    ]
    slider_prep = DataLoaderText2ImageMixin._slider_preparation_modules(prep_mods)
    assert isinstance(slider_prep[0], OptionalRescaleImageChannels)
    assert isinstance(slider_prep[1], OptionalScaleImage)
    assert isinstance(slider_prep[2], OptionalSampleVAEDistribution)

    # Test OptionalNormalizeImageChannels & OptionalImageToVideo in pipelines
    pipeline = LoadingPipeline(
        device=torch.device("cpu"),
        modules=[
            _DictListModule("concept", concepts),
            _DictListModule("target_frames", [1, 1]),
            collect,
            load_pos,
            load_neg,
            load_neg_vid,
            OptionalRescaleImageChannels("negative_image", "negative_image", 0, 1, -1, 1),
            OptionalNormalizeImageChannels("negative_image", "negative_image", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            OutputPipelineModule(["image_path", "image", "negative_image", ("concept.type", "concept_type")]),
        ],
        batch_size=2,
        seed=42,
        state=PipelineState(max_threads=1),
    )
    pipeline.start_next_epoch()
    s0 = next(pipeline)
    s1 = next(pipeline)
    assert s0["negative_image"] is not None
    assert s0["negative_image"].shape == (3, 64, 64)
    # MGDS OutputPipelineModule filters out None values on non-slider samples
    assert "negative_image" not in s1

    # Test OptionalImageToVideo preserves 5D shape and None fallback
    vid_pipeline = LoadingPipeline(
        device=torch.device("cpu"),
        modules=[
            _DictListModule("negative_image", [s0["negative_image"], None]),
            to_vid,
            OutputPipelineModule(["negative_image"]),
        ],
        batch_size=2,
        seed=42,
        state=PipelineState(max_threads=1),
    )
    vid_pipeline.start_next_epoch()
    assert next(vid_pipeline)["negative_image"].shape == (3, 1, 64, 64)
    assert "negative_image" not in next(vid_pipeline)

    # Mixed collation must work in BOTH orderings ([slider, standard] AND [standard, slider])
    collated_pos_first = _slider_safe_collate([s0, s1])
    assert collated_pos_first["image"].shape == (2, 3, 64, 64)
    assert isinstance(collated_pos_first["negative_image"], list)
    assert collated_pos_first["negative_image"][0].shape == (3, 64, 64)
    assert collated_pos_first["negative_image"][1] is None

    collated_std_first = _slider_safe_collate([s1, s0])
    assert collated_std_first["image"].shape == (2, 3, 64, 64)
    assert isinstance(collated_std_first["negative_image"], list)
    assert collated_std_first["negative_image"][0] is None
    assert collated_std_first["negative_image"][1].shape == (3, 64, 64)

    # Validate DataLoaderMgdsMixin forbids SLIDER with FINE_TUNE
    class _StubMgds(DataLoaderMgdsMixin):
        pass

    stub = _StubMgds()
    bad_cfg = TrainConfig.default_values()
    bad_cfg.training_method = TrainingMethod.FINE_TUNE
    c_sl = ConceptConfig.default_values()
    c_sl.type = ConceptType.SLIDER
    c_sl.path = str(pos_dir)
    c_sl.negative_path = str(neg_dir)
    bad_cfg.concepts = [c_sl]
    with pytest.raises(ValueError, match="LORA"):
        stub._create_mgds(bad_cfg, [], TrainProgress(), is_validation=False)


def test_two_pole_trainer_step_and_mixed_sub_batching():
    # Verify __slice_sub_batch distinguishes axis tuple (2,) from per-sample list of length 2
    batch = {
        "latent_image": torch.randn(2, 4, 8, 8),
        "negative_latent_image": [torch.randn(4, 8, 8), None],
        "concept_type": [ConceptType.SLIDER, ConceptType.STANDARD],
        "loss_weight": torch.tensor([1.0, 2.0]),
        "original_resolution": (torch.tensor([64, 64]), torch.tensor([64, 64])),
        "crop_resolution": (torch.tensor([64, 64]), torch.tensor([64, 64])),
        "crop_offset": (torch.tensor([0, 0]), torch.tensor([0, 0])),
    }
    sub_slider = GenericTrainer._GenericTrainer__slice_sub_batch(batch, [0], 2)
    assert sub_slider["latent_image"].shape == (1, 4, 8, 8)
    assert sub_slider["original_resolution"][0].shape == (1,)
    assert sub_slider["negative_latent_image"][0].shape == (4, 8, 8)

    sub_std = GenericTrainer._GenericTrainer__slice_sub_batch(batch, [1], 2)
    assert sub_std["latent_image"].shape == (1, 4, 8, 8)
    assert sub_std["original_resolution"][0].shape == (1,)
    assert sub_std["negative_latent_image"] == [None]

    # Full two-pole gradient & fused back-pass gating verification
    w = nn.Parameter(torch.tensor([1.0]))
    multiplier_history = []
    grad_hook_steps = []
    loss_calls = []

    class _FakeOptimizer(torch.optim.SGD):
        def __init__(self, params, lr=0.1):
            super().__init__(params, lr=lr)
            self.step_calls = 0

        def step_parameter(self, tensor, param_group, i):
            self.step_calls += 1
            grad_hook_steps.append((len(loss_calls), float(tensor.grad.item())))

    class _FakeModel:
        def __init__(self, opt):
            self.model_type = ModelType.FLUX_DEV_1
            self.vae = None
            self.train_dtype = types.SimpleNamespace(torch_dtype=lambda: torch.float32)
            self.autocast_context = nullcontext()
            self.train_progress = TrainProgress()
            self.optimizer = opt
            self.ema = None
            self.multiplier = 1.0

        def set_adapter_multiplier(self, value: float):
            self.multiplier = float(value)
            multiplier_history.append(float(value))

        def evict(self):
            pass

    class _FakeSetup:
        def setup_train_device(self, model, config):
            pass

        def predict(self, model, b, config, train_progress):
            loss_calls.append((model.multiplier, b["latent_image"].clone()))
            pred = w * model.multiplier * b["latent_image"].mean()
            target = torch.zeros_like(pred)
            return {"predicted": pred, "target": target, "loss_type": "target"}

        def calculate_loss(self, model, b, data, config):
            # Signed contrast loss so flipping multiplier + swapping neg/pos gives coherent gradients
            return (data["predicted"] - b["latent_image"].mean()).square().mean()

        def report_to_tensorboard(self, model, config, lr_scheduler, tensorboard):
            pass

        def after_optimizer_step(self, model, config, train_progress):
            pass

    class _FakeDataLoader:
        def __init__(self, batches):
            self.batches = batches

        def get_data_set(self):
            return types.SimpleNamespace(
                start_next_epoch=lambda: None,
                approximate_length=lambda: len(self.batches),
            )

        def get_data_loader(self):
            return self.batches

    opt = _FakeOptimizer([w], lr=0.1)
    trainer = GenericTrainer.__new__(GenericTrainer)
    TimedActionMixin.__init__(trainer)
    trainer.model = _FakeModel(opt)
    trainer.model_setup = _FakeSetup()
    trainer.config = TrainConfig.default_values()
    trainer.config.training_method = TrainingMethod.LORA
    trainer.config.train_dtype = DataType.FLOAT_32
    trainer.config.epochs = 1
    trainer.config.batch_size = 2
    trainer.config.slider_diff_weight = 0.0
    trainer.config.gradient_accumulation_steps = 1
    trainer.config.only_cache = False
    trainer.config.validation = False
    trainer.config.sample_after = 0
    trainer.config.sample_after_unit = TimeUnit.NEVER
    trainer.config.backup_after = 0
    trainer.config.backup_after_unit = TimeUnit.NEVER
    trainer.config.save_every = 0
    trainer.config.save_every_unit = TimeUnit.NEVER
    trainer.config.validate_after_unit = TimeUnit.NEVER
    trainer.config.debug_mode = False
    trainer.config.tensorboard = False
    trainer.config.multi_gpu = False
    trainer.config.fused_gradient_reduce = False
    trainer.config.optimizer.fused_back_pass = True
    trainer.config.optimizer.optimizer = types.SimpleNamespace(
        is_schedule_free=False,
        supports_fused_back_pass=lambda: True,
    )
    trainer.parameters = [w]
    trainer.grad_hook_handles = []
    trainer.sample_queue = []
    trainer.one_step_trained = False
    trainer.commands = types.SimpleNamespace(
        get_stop_command=lambda: False,
        get_and_reset_sample_default_command=lambda: False,
        get_and_reset_sample_custom_commands=list,
        get_and_reset_backup_command=lambda: False,
        get_and_reset_save_command=lambda: False,
        backup=lambda: None,
        save=lambda: None,
    )
    trainer.callbacks = types.SimpleNamespace(
        on_update_status=lambda _msg: None,
        on_update_train_progress=lambda *_a: None,
        on_sample_default=lambda *_a: None,
        on_sample_custom=lambda *_a: None,
        on_update_sample_default_progress=lambda *_a: None,
        on_update_sample_custom_progress=lambda *_a: None,
    )
    trainer.tensorboard = types.SimpleNamespace(add_scalar=lambda *_a: None, close=lambda: None)
    trainer.train_progress_tracker = {}
    trainer._GenericTrainer__suppress_fused_back_pass = False

    mixed_batch = {
        "latent_image": torch.tensor([[[[2.0]]], [[[3.0]]]]),
        "negative_latent_image": [torch.tensor([[[0.5]]]), None],
        "concept_type": [ConceptType.SLIDER, ConceptType.STANDARD],
        "loss_weight": torch.tensor([1.0, 1.0]),
    }
    trainer.data_loader = _FakeDataLoader([mixed_batch])

    with (
        patch("modules.trainer.GenericTrainer.create.create_lr_scheduler", return_value=types.SimpleNamespace(step=lambda: None)),
        patch("modules.trainer.GenericTrainer.multi.is_master", return_value=False),
        patch("modules.trainer.GenericTrainer.multi.sync_commands", lambda _c: None),
        patch("modules.trainer.GenericTrainer.multi.reduce_tensor_mean", lambda _t: None),
        patch("modules.trainer.GenericTrainer.multi.reduce_grads_mean", lambda *_a: None),
    ):
        trainer.train()

    for h in trainer.grad_hook_handles:
        h.remove()

    # Non-slider ran at default 1.0, then slider pole +1, slider pole -1, then finally reset to 1.0
    assert multiplier_history == [1.0, -1.0, 1.0]
    assert len(loss_calls) == 3
    # Non-slider sub-batch used latent=3.0 (m=1.0), slider +1 used 2.0, slider -1 used 0.5
    assert pytest.approx(float(loss_calls[0][1].item())) == 3.0
    assert pytest.approx(float(loss_calls[1][1].item())) == 2.0
    assert pytest.approx(float(loss_calls[2][1].item())) == 0.5
    # Fused back pass hook must fire ONCE after the final negative pole (3 forward calls completed), NOT mid-step!
    assert len(grad_hook_steps) == 1
    assert grad_hook_steps[0][0] == 3
    assert trainer.model.multiplier == 1.0


def test_slider_sampling_stitching_and_staged_samplers(tmp_path: Path):
    img_neg = Image.new("RGB", (64, 64), (10, 20, 30))
    img_pos = Image.new("RGB", (64, 64), (200, 210, 220))
    stitched = stitch_slider_images([img_neg, img_pos], multipliers=(-1.0, 1.0))
    assert stitched.size[0] == 64 * 2 + 8
    assert stitched.size[1] > 64

    vid_neg = torch.zeros(3, 32, 32, 3, dtype=torch.uint8)
    vid_pos = torch.ones(3, 32, 32, 3, dtype=torch.uint8)
    vid_stitched = stitch_slider_videos([vid_neg, vid_pos])
    assert vid_stitched.shape == (3, 32, 64, 3)

    # Verify BaseModelSampler.sample_slider_all resolves identical seed, suppresses intermediate saves, saves once
    class _ToySampler(BaseModelSampler):
        def __init__(self, model):
            super().__init__(torch.device("cpu"), torch.device("cpu"))
            self.model = model
            self.seen = []

        def sample(self, sample_config, destination, image_format, video_format, audio_format, on_sample=..., on_update_progress=...):
            self.seen.append((sample_config.seed, sample_config.slider_multiplier, self.model.multiplier))
            color = (20, 20, 20) if self.model.multiplier < 0 else (230, 230, 230)
            out = ModelSamplerOutput(FileType.IMAGE, Image.new("RGB", (32, 32), color))
            self.save_sampler_output(out, destination, image_format, video_format, audio_format)
            on_sample(out)

    class _SamplerModel:
        NATIVE_FPS = 24
        def __init__(self):
            self.multiplier = 1.0
        def set_adapter_multiplier(self, v):
            self.multiplier = v

    m = _SamplerModel()
    sampler = _ToySampler(m)
    sc = SampleConfig.default_values()
    sc.random_seed = True
    dest = tmp_path / "out" / "sample_1"
    outs = sampler.sample_slider_all(
        [sc],
        [str(dest)],
        image_format=ImageFormat.PNG,
        multipliers=(-1.0, 1.0),
    )
    assert len(outs) == 1
    assert outs[0].file_type == FileType.IMAGE
    assert len(sampler.seen) == 2
    assert sampler.seen[0][0] == sampler.seen[1][0]
    assert sampler.seen[0][2] == -1.0
    assert sampler.seen[1][2] == 1.0
    assert m.multiplier == 1.0
    assert Path(str(dest) + ".png").is_file()

    # Check staged sampler classes expose __denoise with multiplier handling
    for sampler_cls in (AnimaSampler, Flux2Sampler, Krea2Sampler, MossSampler, QwenSampler, ZImageSampler):
        src = sampler_cls._AnimaSampler__denoise if sampler_cls is AnimaSampler else getattr(
            sampler_cls, f"_{sampler_cls.__name__}__denoise"
        )
        assert callable(src)


def test_transformer_override_and_moss_single_file(tmp_path: Path):
    ctrl = ModelTabController(TrainConfig.default_values())
    for mt in (
        ModelType.KREA_2,
        ModelType.MOSS,
        ModelType.FLUX_2,
        ModelType.Z_IMAGE,
        ModelType.ERNIE,
        ModelType.CHROMA_1,
        ModelType.QWEN,
        ModelType.ANIMA,
        ModelType.HUNYUAN_VIDEO,
    ):
        ctrl.train_config.model_type = mt
        assert ctrl.supports_override_transformer() is True

    # Test MOSS checkpoint conversion with bare + ComfyUI diffusion_model prefixes + quant_type retention
    class _FakeGGUF(torch.Tensor):
        @staticmethod
        def __new__(cls, data, quant_type="Q4_K"):
            inst = torch.Tensor._make_subclass(cls, data)
            inst.quant_type = quant_type
            return inst

    raw_ckpt = {
        "model.diffusion_model.layers.0.proj.qkv.weight": _FakeGGUF(torch.randn(12, 4), quant_type="Q8_0"),
        "model.diffusion_model.layers.0.comfy_attention.config": torch.tensor([1]),
        "diffusion_model.video_embedder.weight": torch.randn(4, 4),
    }
    converted = _convert_moss_image_transformer_checkpoint(raw_ckpt)
    assert "layers.0.proj.q.weight" in converted
    assert "layers.0.proj.k.weight" in converted
    assert "layers.0.proj.v.weight" in converted
    assert "video_embedder.weight" in converted
    assert not any("comfy_attention" in k for k in converted)
    assert getattr(converted["layers.0.proj.q.weight"], "quant_type", None) == "Q8_0"

    # Full MOSSImageTransformer.from_single_file roundtrip with tiny config + prefixed fused qkv safetensors
    tiny_cfg = {"dim": 16, "ffn_dim": 32, "num_heads": 2, "num_layers": 1, "text_dim": 8, "freq_dim": 8, "latent_channels": 8, "rope_base": 10000.0}
    cfg_dir = tmp_path / "moss_base" / "transformer"
    cfg_dir.mkdir(parents=True)
    tiny_model = MOSSImageTransformer(**tiny_cfg)
    (cfg_dir / "config.json").write_text(json.dumps(tiny_cfg))

    sd = {}
    for k, v in tiny_model.state_dict().items():
        if k.startswith("layers.0.proj.q."):
            continue
        if k.startswith("layers.0.proj.k."):
            continue
        if k.startswith("layers.0.proj.v."):
            continue
        sd["diffusion_model." + k] = v.clone()
    q = tiny_model.state_dict()["layers.0.proj.q.weight"]
    k = tiny_model.state_dict()["layers.0.proj.k.weight"]
    v = tiny_model.state_dict()["layers.0.proj.v.weight"]
    sd["diffusion_model.layers.0.proj.qkv.weight"] = torch.cat([q, k, v], dim=0).clone()

    single_file = tmp_path / "moss_override.safetensors"
    save_file(sd, str(single_file))

    weight_dtypes = TrainConfig.default_values().weight_dtypes()
    weight_dtypes.train_dtype = DataType.FLOAT_32
    weight_dtypes.fallback_train_dtype = DataType.FLOAT_32
    weight_dtypes.transformer = DataType.FLOAT_32

    loader = HFModelLoaderMixin()
    loaded, mat_fn = loader._load_transformer(
        MOSSImageTransformer,
        weight_dtypes,
        base_model_name=str(tmp_path / "moss_base"),
        transformer_model_name=str(single_file),
        quantization=TrainConfig.default_values().quantization,
        config=str(tmp_path / "moss_base"),
        stream_from_disk=False,
    )
    assert mat_fn is None
    assert isinstance(loaded, MOSSImageTransformer)
    assert not loaded.layers[0].proj["q"].weight.is_meta

    # Test HFModelLoaderMixin._load_transformer with local diffusers directory
    diffusers_dir = tmp_path / "local_transformer"
    tiny_model.save_pretrained(str(diffusers_dir))
    loaded_dir, _ = loader._load_transformer(
        MOSSImageTransformer,
        weight_dtypes,
        base_model_name=str(tmp_path / "moss_base"),
        transformer_model_name=str(diffusers_dir),
        quantization=TrainConfig.default_values().quantization,
        config=str(tmp_path / "moss_base"),
        stream_from_disk=False,
    )
    assert isinstance(loaded_dir, MOSSImageTransformer)


def test_ui_filter_and_train_controller_slider_validation(tmp_path: Path):
    assert "SLIDER" in BaseConceptTabView._FILTER_TYPES

    pos = tmp_path / "pos"
    neg = tmp_path / "neg"
    pos.mkdir()
    neg.mkdir()

    cfg = TrainConfig.default_values()
    cfg.training_method = TrainingMethod.FINE_TUNE
    c = ConceptConfig.default_values()
    c.name = "slider_test"
    c.type = ConceptType.SLIDER
    c.path = str(pos)
    c.negative_path = ""
    cfg.concepts = [c]

    errors_seen = []
    ctrl = TrainUIController.__new__(TrainUIController)
    ctrl.training_thread = None
    ctrl.train_config = cfg
    ctrl._local_gpu_count = 1
    ctrl.view = types.SimpleNamespace(show_validation_errors=lambda errs: errors_seen.append(errs))

    with patch("modules.ui.TrainUIController.flush_and_validate_all", return_value=[]):
        ctrl.start_training()

    assert len(errors_seen) == 1
    assert any("LORA" in msg for msg in errors_seen[0])
    assert any("Negative Path (-1) is empty" in msg for msg in errors_seen[0])

    # Now LORA + valid non-empty dirs with 0 matching pairs -> reports 0 pairs
    cfg.training_method = TrainingMethod.LORA
    c.negative_path = str(neg)
    errors_seen.clear()
    with patch("modules.ui.TrainUIController.flush_and_validate_all", return_value=[]):
        ctrl.start_training()

    assert len(errors_seen) == 1
    assert any("No matching pairs" in msg for msg in errors_seen[0])

    # Verify ConceptWindowController.validate_slider_pairs on valid pairs
    _save_png(pos / "pair1.png", (64, 64), (255, 0, 0))
    _save_png(neg / "pair1.png", (64, 64), (0, 0, 255))
    c_ctrl = ConceptWindowController(TrainConfig.default_values(), c)
    validation_seen = []
    mock_view = types.SimpleNamespace(
        concept_stats_tab=object(),
        components=types.SimpleNamespace(call_after=lambda _tab, _ms, fn: fn()),
        _disable_scan_buttons=lambda: None,
        _enable_scan_buttons=lambda: None,
        _update_concept_stats=lambda _c: None,
        _show_slider_validation_result=lambda res: validation_seen.append(res),
    )
    c_ctrl.validate_slider_pairs(mock_view)
    assert len(validation_seen) == 1
    assert validation_seen[0]["valid"] is True
    assert validation_seen[0]["matching_pairs"] == 1


def _run_load_image_pipeline(image_path: Path, *, range_min=0.0, range_max=1.0, dtype=None):
    """Run a one-image LoadingPipeline through OneTrainer's LoadImage and return the output tensor."""
    pipeline = LoadingPipeline(
        device=torch.device("cpu"),
        modules=[
            _DictListModule("image_path", [str(image_path)]),
            LoadImage("image_path", "image", range_min, range_max, {".png", ".jpg"}, dtype=dtype),
            OutputPipelineModule(["image"]),
        ],
        batch_size=1,
        seed=1,
        state=PipelineState(max_threads=1),
    )
    pipeline.start_next_epoch()
    return next(pipeline)["image"]


def test_load_image_preserves_image_range_and_avoids_double_normalization(tmp_path: Path):
    """Regression test for the doubled /255 in the pinned mgds LoadImage (mgds #63 / OneTrainer #1790).

    With the duplicated normalization every image came out as [0, 1/255], which RescaleImageChannels
    then mapped to [-1.0, -0.992]: all training images (slider +1 and -1 poles included) reached the
    VAE as solid black, so a Slider LoRA could only learn from identical black latents.
    """
    # torchvision path (the normal path for png/jpg)
    _save_png(tmp_path / "rgb.png", (8, 8), (240, 20, 120))
    loaded = _run_load_image_pipeline(tmp_path / "rgb.png")
    assert loaded.shape == (3, 8, 8)
    assert loaded.dtype == torch.float32
    # values must be the real 8-bit values scaled once into [0, 1]
    torch.testing.assert_close(loaded[:, 0, 0], torch.tensor([240.0, 20.0, 120.0]) / 255.0, rtol=0, atol=1e-6)
    assert loaded.min() >= 0.0
    assert loaded.max() > 0.9

    # range scaling is applied exactly once, not twice
    _save_png(tmp_path / "white.png", (4, 4), (255, 255, 255))
    white = _run_load_image_pipeline(tmp_path / "white.png", range_min=-1.0, range_max=1.0)
    assert float(white.max()) == pytest.approx(1.0)
    _save_png(tmp_path / "black.png", (4, 4), (0, 0, 0))
    black = _run_load_image_pipeline(tmp_path / "black.png", range_min=-1.0, range_max=1.0)
    assert float(black.min()) == pytest.approx(-1.0)

    # a mid gray image must stay mid gray instead of collapsing into black
    _save_png(tmp_path / "gray.png", (4, 4), (128, 128, 128))
    gray = _run_load_image_pipeline(tmp_path / "gray.png", range_min=-1.0, range_max=1.0)
    assert float(gray.mean()) == pytest.approx(128.0 / 255.0 * 2.0 - 1.0, abs=2e-3)

    # a downscaled dtype (bfloat16) keeps the same range
    bf16_loaded = _run_load_image_pipeline(tmp_path / "rgb.png", dtype=torch.bfloat16)
    assert bf16_loaded.dtype == torch.bfloat16
    torch.testing.assert_close(bf16_loaded, loaded.to(dtype=torch.bfloat16), rtol=0, atol=0)


def test_load_image_pillow_fallback_matches_torchvision(tmp_path: Path, monkeypatch):
    """The Pillow fallback had the same duplicated normalization plus a repeated convert/ToTensor."""
    _save_png(tmp_path / "rgb.png", (8, 8), (240, 20, 120))
    torchvision_result = _run_load_image_pipeline(tmp_path / "rgb.png")

    def _raise_runtime_error(*_args, **_kwargs):
        # the loader falls back to Pillow on RuntimeError from the torchvision decoder
        raise RuntimeError("forced pillow fallback")

    monkeypatch.setattr(load_image_module, "read_image", _raise_runtime_error)
    pillow_result = _run_load_image_pipeline(tmp_path / "rgb.png")
    torch.testing.assert_close(pillow_result, torchvision_result, rtol=0, atol=0)


def test_load_image_reads_16bit_images_as_8bit_range(tmp_path: Path):
    """16-bit samples are normalized by their dtype max (mgds #55), then scaled once."""
    data = np.arange(16, dtype=np.uint16).reshape(4, 4) * 4000
    image = Image.fromarray(data)
    assert image.mode == "I;16"
    path = tmp_path / "gray16.png"
    image.save(path)

    loaded = _run_load_image_pipeline(path)
    assert loaded.shape == (3, 4, 4)
    assert float(loaded.min()) == pytest.approx(0.0)
    assert float(loaded.max()) == pytest.approx(float(data.max()) / 65535.0, abs=1e-6)


def test_slider_pos_and_neg_latents_differ_after_vae_encoding(tmp_path: Path):
    """End-to-end guard for the bug: distinct +1/-1 images must produce distinct VAE latents.

    With the doubled /255 both images were already identical black before the VAE, so the latent
    difference was zero. The slider diff weight then had nothing to weight, and both poles of the
    LoRA were trained on the same (black) target.
    """
    pos_path = tmp_path / "pos.png"
    neg_path = tmp_path / "neg.png"
    _save_png(pos_path, (64, 64), (240, 20, 20))
    _save_png(neg_path, (64, 64), (20, 20, 240))

    class _DeterministicDistribution:
        # carries the "encoded" tensor straight through the "mean" sampling mode
        def __init__(self, mean):
            self._mean = mean

        def mode(self):
            return self._mean

        def sample(self):
            return self._mean

    class _IdentityVAE:
        def encode(self, x):
            # a deterministic "encoder" that keeps the pixel statistics visible in the latent
            return types.SimpleNamespace(latent_dist=_DeterministicDistribution(x))

    def encode(image_path: Path) -> torch.Tensor:
        pipeline = LoadingPipeline(
            device=torch.device("cpu"),
            modules=[
                _DictListModule("image_path", [str(image_path)]),
                LoadImage("image_path", "image", 0, 1, {".png"}),
                RescaleImageChannels("image", "image", 0, 1, -1, 1),
                OptionalEncodeVAE(
                    in_name="image",
                    out_name="latent_image_distribution",
                    vae=_IdentityVAE(),
                    autocast_contexts=[nullcontext()],
                    dtype=torch.float32,
                ),
                OptionalSampleVAEDistribution("latent_image_distribution", "latent_image", "mean"),
                OutputPipelineModule(["latent_image"]),
            ],
            batch_size=1,
            seed=1,
            state=PipelineState(max_threads=1),
        )
        pipeline.start_next_epoch()
        return next(pipeline)["latent_image"]

    pos_latent = encode(pos_path)
    neg_latent = encode(neg_path)

    # the bug collapsed both training images into the single, all-but-black range [-1.0, -0.992]
    for latent in (pos_latent, neg_latent):
        assert float(latent.min()) < -0.5, "image collapsed into the near-black range [-1.0, -0.992]"
        assert float(latent.max()) > 0.5, "image collapsed into the near-black range [-1.0, -0.992]"
    assert float((pos_latent - neg_latent).abs().mean()) > 0.1

    # and the slider diff weighting sees a real (non-degenerate) difference map
    weight = compute_slider_diff_weight(
        types.SimpleNamespace(model_type=None, vae=None),
        pos_latent.unsqueeze(0),
        neg_latent.unsqueeze(0),
        1.0,
    )
    assert weight is not None
    assert weight.shape[-2:] == (64, 64)
    assert float(weight.mean()) == pytest.approx(1.0, abs=1e-5)


def test_stale_slider_cache_warning(tmp_path: Path, capsys):
    """A cache filled while the loader bug was live must not be reused silently."""
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = TrainConfig.default_values()
    trainer.config.cache_dir = str(tmp_path)
    trainer.config.latent_caching = True

    # no slider concept -> nothing to warn about
    trainer._GenericTrainer__warn_if_stale_slider_cache()
    assert "[Slider]" not in capsys.readouterr().out

    slider = ConceptConfig.default_values()
    slider.name = "age"
    slider.type = ConceptType.SLIDER
    slider.path = str(tmp_path / "pos")
    slider.negative_path = str(tmp_path / "neg")
    trainer.config.concepts = [slider]

    # no cache yet -> nothing to warn about
    trainer._GenericTrainer__warn_if_stale_slider_cache()
    assert "[Slider]" not in capsys.readouterr().out

    # an empty cache root -> nothing to warn about yet
    (tmp_path / "image").mkdir()
    trainer._GenericTrainer__warn_if_stale_slider_cache()
    assert "[Slider]" not in capsys.readouterr().out

    # a populated cache -> warn, naming the directory and the fix
    (tmp_path / "image" / "epoch-0").mkdir()
    trainer._GenericTrainer__warn_if_stale_slider_cache()
    out = capsys.readouterr().out
    assert "WARNING: reusing the existing latent cache" in out
    assert str(tmp_path / "image") in out
    assert "Clear Cache Before Training" in out
