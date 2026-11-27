import os
import time

from modules.dataLoader.LoadImage import LoadImage
from modules.dataLoader.wuerstchen.EncodeWuerstchenEffnet import EncodeWuerstchenEffnet
from modules.util.enum.ConceptType import ConceptType
from modules.util.slider_util import log_slider_scan_result, scan_slider_pairs

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.ImageToVideo import ImageToVideo
from mgds.pipelineModules.LoadVideo import LoadVideo
from mgds.pipelineModules.NormalizeImageChannels import NormalizeImageChannels
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from tqdm import tqdm


class CollectSliderPaths(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            concept_in_name: str,
            path_in_name: str,
            neg_path_in_name: str,
            type_in_name: str,
            include_subdirectories_in_name: str,
            enabled_in_name: str,
            path_out_name: str,
            neg_path_out_name: str,
            concept_out_name: str,
            extensions: set[str] | list[str],
            include_postfix: list[str] | None,
            exclude_postfix: list[str] | None,
    ):
        super().__init__()
        self.concept_in_name = concept_in_name
        self.path_in_name = path_in_name
        self.neg_path_in_name = neg_path_in_name
        self.type_in_name = type_in_name
        self.include_subdirectories_in_name = include_subdirectories_in_name
        self.enabled_in_name = enabled_in_name

        self.path_out_name = path_out_name
        self.neg_path_out_name = neg_path_out_name
        self.concept_out_name = concept_out_name

        self.extensions = {ext.lower() for ext in extensions}
        self.include_postfix = include_postfix
        self.exclude_postfix = tuple(exclude_postfix or ())

        self.paths: list[str] = []
        self.negative_paths: list[str | None] = []
        self.concepts: list[dict] = []

    def length(self) -> int:
        return len(self.paths)

    def get_inputs(self) -> list[str]:
        return [self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.path_out_name, self.neg_path_out_name, self.concept_out_name]

    @staticmethod
    def __list_files(path: str, include_subdirectories: bool) -> list[str]:
        dir_list = [os.path.join(path, filename) for filename in os.listdir(path)]
        files = list(filter(os.path.isfile, dir_list))
        if include_subdirectories:
            for sub_directory in filter(os.path.isdir, dir_list):
                if os.path.basename(sub_directory).startswith("."):
                    continue
                files.extend(CollectSliderPaths.__list_files(sub_directory, include_subdirectories))
        return files

    def start(self, variation: int):
        self.paths = []
        self.negative_paths = []
        self.concepts = []

        for in_index in tqdm(range(self._get_previous_length(self.concept_in_name)), desc="enumerating sample paths"):
            concept = self._get_previous_item(variation, self.concept_in_name, in_index)
            if not concept.get(self.enabled_in_name, True):
                continue

            include_subdirectories = bool(self._get_previous_item(variation, self.include_subdirectories_in_name, in_index))
            pos_path = concept[self.path_in_name]
            concept_type = ConceptType(concept.get(self.type_in_name, ConceptType.STANDARD))

            if concept_type == ConceptType.SLIDER:
                neg_path = concept.get(self.neg_path_in_name, "")
                concept_label = concept.get("name") or pos_path
                if not pos_path or not os.path.isdir(pos_path):
                    raise ValueError(f"Slider concept '{concept_label}' has invalid positive (+1) path: {pos_path!r}")
                if not neg_path or not os.path.isdir(neg_path):
                    raise ValueError(f"Slider concept '{concept_label}' has invalid negative (-1) path: {neg_path!r}")

                scan_result = scan_slider_pairs(
                    pos_dir=pos_path,
                    neg_dir=neg_path,
                    include_subdirectories=include_subdirectories,
                    extensions=self.extensions,
                    exclude_postfix=self.exclude_postfix,
                    check_dimensions=True,
                )
                log_slider_scan_result(concept_label, scan_result)
                if scan_result.matching_pairs_count == 0:
                    print(f"[Slider] WARNING: Slider concept '{concept_label}' has 0 valid matching pairs.")

                for pair in scan_result.matched_pairs:
                    self.paths.append(pair.pos_path)
                    self.negative_paths.append(pair.neg_path)
                    self.concepts.append(concept)
            else:
                file_names = sorted(self.__list_files(pos_path, include_subdirectories))
                file_names = [name for name in file_names if os.path.splitext(name)[1].lower() in self.extensions]
                if self.include_postfix:
                    file_names = [
                        name for name in file_names
                        if any(os.path.splitext(name)[0].endswith(postfix) for postfix in self.include_postfix)
                    ]
                if self.exclude_postfix:
                    file_names = [
                        name for name in file_names
                        if not any(os.path.splitext(name)[0].endswith(postfix) for postfix in self.exclude_postfix)
                    ]
                self.paths.extend(file_names)
                self.negative_paths.extend([None] * len(file_names))
                self.concepts.extend([concept] * len(file_names))

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            self.path_out_name: self.paths[index],
            self.neg_path_out_name: self.negative_paths[index],
            self.concept_out_name: self.concepts[index],
        }


class OptionalLoadImage(LoadImage):
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)
        if path is None:
            return {self.image_out_name: None}
        return super().get_item(variation, index, requested_name)


class OptionalLoadVideo(LoadVideo):
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)
        if path is None:
            return {self.video_out_name: None}
        return super().get_item(variation, index, requested_name)


class OptionalImageToVideo(ImageToVideo):
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tensor = self._get_previous_item(variation, self.in_name, index)
        if tensor is None:
            return {self.out_name: None}
        return super().get_item(variation, index, requested_name)


class OptionalRescaleImageChannels(RescaleImageChannels):
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.image_in_name, index)
        if image is None:
            return {self.image_out_name: None}
        return super().get_item(variation, index, requested_name)


class OptionalScaleImage(ScaleImage):
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index)
        if image is None:
            return {self.out_name: None}
        return super().get_item(variation, index, requested_name)


class OptionalNormalizeImageChannels(NormalizeImageChannels):
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.image_in_name, index)
        if image is None:
            return {self.image_out_name: None}
        return super().get_item(variation, index, requested_name)


class OptionalEncodeVAE(EncodeVAE):
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index)
        if image is None:
            return {self.out_name: None}
        return super().get_item(variation, index, requested_name)


class OptionalSampleVAEDistribution(SampleVAEDistribution):
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        distribution = self._get_previous_item(variation, self.in_name, index)
        if distribution is None:
            return {self.out_name: None}
        return super().get_item(variation, index, requested_name)


class OptionalEncodeWuerstchenEffnet(EncodeWuerstchenEffnet):
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index)
        if image is None:
            return {self.out_name: None}
        return super().get_item(variation, index, requested_name)


class VerboseDiskCache(DiskCache):
    def __init__(self, *args, cache_label: str = "disk", **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_label = cache_label

    def start(self, out_variation: int):
        t0 = time.perf_counter()
        print(f"[Slider][Cache] Preparing {self.cache_label} cache (epoch {out_variation}) at '{self.cache_dir}'...")
        super().start(out_variation)
        elapsed = time.perf_counter() - t0
        total_items = sum(self.group_output_samples.values())
        print(
            f"[Slider][Cache] Ready {self.cache_label} cache (epoch {out_variation}): "
            f"{total_items} item(s) across {len(self.group_output_samples)} group(s) in {elapsed:.2f}s."
        )
