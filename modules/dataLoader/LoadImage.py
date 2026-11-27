import os.path

from mgds.PipelineModule import SkipSampleError
from mgds.pipelineModules.LoadImage import LoadImage as MGDSLoadImage

import torch
from torchvision import transforms
from torchvision.io import read_image

import numpy as np
from PIL import Image, ImageOps


class LoadImage(MGDSLoadImage):
    """mgds LoadImage with the double normalization of the pinned mgds commit repaired.

    mgds a5f35a3e ("Bug fixes" #63) added dtype-max normalization plus a range scaling step before
    the already existing "Transform 0 - 255 to 0-1" block, but did not remove that block. Every image
    is therefore divided by 255 twice and has (range_max - range_min) + range_min applied twice: an
    image in [0, 1] comes out as [0, 1/255] and is then rescaled to [-1.0, -0.992] instead of
    [-1.0, 1.0], so all training images reach the VAE as (near) solid black.

    The Pillow fallback repeats the same range scaling, which is a no-op for the [0, 1] range the
    training pipeline asks for but not for e.g. the [-1, 1] range of the VAE fine-tune loader
    ([-1, 1] becomes [-3, 1]). It also calls convert() before checking for 16-bit input, so its
    16-bit branch can never run.

    This override applies the range scaling exactly once for both the torchvision and the Pillow
    path, so it matches the behaviour of mgds before #63. Delete it once the pinned mgds commit no
    longer contains the duplicated steps in `get_item` / `_load_pillow`.
    """

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)

        ext = os.path.splitext(path)[1]
        if ext.lower() in self.supported_extensions:
            try:
                image_tensor = read_image(path, self.mode, apply_exif_orientation=True) \
                    .to(device=self.pipeline.device)

                # Normalize integer dtypes (e.g. uint8 -> 255, uint16 -> 65535) in float32 before
                # casting to self.dtype so uint16 max (65535) does not overflow float16 (65504).
                if image_tensor.is_floating_point():
                    image_tensor = image_tensor.to(dtype=torch.float32)
                else:
                    image_max = torch.iinfo(image_tensor.dtype).max
                    image_tensor = image_tensor.to(dtype=torch.float32) / image_max

                image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min

                if self.dtype:
                    image_tensor = image_tensor.to(dtype=self.dtype)
            except FileNotFoundError:
                image_tensor = None
            except RuntimeError:
                # Torch builtins couldn't load it because not all image formats and
                # variations there are supported. Fall back to pillow.
                image_tensor = self._load_pillow(path)
            except Exception as e:
                if self.skip_corrupt:
                    print("could not load image, it might be corrupted, skipping: " + path)
                    raise SkipSampleError(f"could not load image: {path}") from e
                print("could not load image, it might be corrupted: " + path)
                raise
        else:
            image_tensor = None

        return {
            self.image_out_name: image_tensor
        }

    def _load_pillow(self, path):
        """Fallback method to load the image through pillow.

        MUCH slower, because the tensor transform holds the GIL.
        """
        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image)

            t = transforms.ToTensor()
            if image.mode.startswith("I;16"):
                # PIL's convert('L'/'RGB') clamps 16-bit grayscale values >= 255, and ToTensor()
                # reinterprets I;16 as signed int16 (wrapping >=32768 negative). Read the raw 16-bit
                # samples instead.
                image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)
                image_tensor = image_tensor.to(device=self.pipeline.device) / 65535.0
                if self.pillow_mode == 'RGB':
                    image_tensor = image_tensor.repeat(3, 1, 1)
            else:
                image = image.convert(self.pillow_mode)
                image_tensor = t(image).to(device=self.pipeline.device, dtype=torch.float32)

            image_tensor = image_tensor * (self.range_max - self.range_min) + self.range_min

            if self.dtype:
                image_tensor = image_tensor.to(dtype=self.dtype)
        except FileNotFoundError:
            image_tensor = None
        except Exception as e:
            if self.skip_corrupt:
                print("could not load image, it might be corrupted, skipping: " + path)
                raise SkipSampleError(f"could not load image: {path}") from e
            print("could not load image, it might be corrupted: " + path)
            raise

        return image_tensor
