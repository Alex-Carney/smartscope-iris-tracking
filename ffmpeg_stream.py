import asyncio
from typing import List, Optional

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"

class FFMPEGMJPEGStream:
    def __init__(self, device_name: str, width: int, height: int, fps: int):
        self.device_name = device_name
        self.width = width
        self.height = height
        self.fps = fps
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.buf = bytearray()

    def _build_cmd(self) -> List[str]:
        return [
            "ffmpeg",
            "-f", "dshow",
            "-vsync", "drop",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-vcodec", "mjpeg",
            "-i", f"video={self.device_name}",
            "-f", "mjpeg",
            "pipe:1",
        ]

    async def start(self) -> None:
        cmd = self._build_cmd()
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def read_jpeg(self) -> Optional[bytes]:
        if not self.proc or not self.proc.stdout:
            return None
        read = self.proc.stdout.read
        while True:
            chunk = await read(65536)  # bigger read (64 KiB)
            if not chunk:
                return None
            self.buf.extend(chunk)
            # find SOI/EOI
            s = self.buf.find(SOI)
            if s < 0:
                # keep a small tail in case SOI crosses chunk boundary
                if len(self.buf) > 2048:
                    del self.buf[:-2048]
                continue
            e = self.buf.find(EOI, s + 2)
            if e < 0:
                # keep from SOI onward
                if s > 0:
                    del self.buf[:s]
                continue
            # slice complete JPEG
            e2 = e + 2
            jpg = bytes(self.buf[s:e2])
            del self.buf[:e2]
            return jpg

    async def stop(self) -> None:
        try:
            if self.proc and self.proc.returncode is None:
                self.proc.terminate()
        except Exception:
            pass