# ffmpeg_stream.py
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
        self.buf = b""

    def _build_cmd(self) -> List[str]:
        return [
            "ffmpeg",
            "-f", "dshow",
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
        while True:
            chunk = await self.proc.stdout.read(4096)
            if not chunk:
                return None
            self.buf += chunk
            s = self.buf.find(SOI)
            if s < 0:
                self.buf = self.buf[-1024:]
                continue
            e = self.buf.find(EOI, s + 2)
            if e < 0:
                self.buf = self.buf[s:]
                continue
            jpg = self.buf[s:e+2]
            self.buf = self.buf[e+2:]
            return jpg

    async def stop(self) -> None:
        """Gracefully stop ffmpeg and fully drain pipes to avoid Proactor warnings on Windows."""
        try:
            if self.proc and self.proc.returncode is None:
                # Ask ffmpeg to exit
                self.proc.terminate()
                try:
                    # Drain stdout/stderr to EOF so transports close cleanly
                    await asyncio.wait_for(self.proc.communicate(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Be forceful if it hangs
                    self.proc.kill()
                    try:
                        await self.proc.communicate()
                    except Exception:
                        pass
        finally:
            self.proc = None
            self.buf = b""
