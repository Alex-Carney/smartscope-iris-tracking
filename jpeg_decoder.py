from turbojpeg import TurboJPEG, TJPF_BGR
from typing import Optional
import numpy as np

class JPEGDecoder:
    def __init__(self, lib_path: Optional[str] = None):
        self.jpeg = TurboJPEG(lib_path) if lib_path else TurboJPEG()

    def decode_bgr(self, jpeg_bytes: bytes) -> np.ndarray:
        return self.jpeg.decode(jpeg_bytes, pixel_format=TJPF_BGR)