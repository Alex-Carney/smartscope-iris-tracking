# import asyncio
# from typing import List, Optional

# SOI = b"\xff\xd8"
# EOI = b"\xff\xd9"

# class FFMPEGMJPEGStream:
#     def __init__(self, device_name: str, width: int, height: int, fps: int):
#         self.device_name = device_name
#         self.width = width
#         self.height = height
#         self.fps = fps
#         self.proc: Optional[asyncio.subprocess.Process] = None
#         self.buf = bytearray()

#     def _build_cmd(self) -> List[str]:
#         return [
#             "ffmpeg",
#             "-f", "dshow",
#             "-vsync", "drop",
#             "-video_size", f"{self.width}x{self.height}",
#             "-framerate", str(self.fps),
#             "-vcodec", "mjpeg",
#             "-i", f"video={self.device_name}",
#             "-f", "mjpeg",
#             "pipe:1",
#         ]

#     async def start(self) -> None:
#         cmd = self._build_cmd()
#         self.proc = await asyncio.create_subprocess_exec(
#             *cmd,
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.DEVNULL,
#         )

#     async def read_jpeg(self) -> Optional[bytes]:
#         if not self.proc or not self.proc.stdout:
#             return None
#         read = self.proc.stdout.read
#         while True:
#             chunk = await read(65536)  # bigger read (64 KiB)
#             if not chunk:
#                 return None
#             self.buf.extend(chunk)
#             # find SOI/EOI
#             s = self.buf.find(SOI)
#             if s < 0:
#                 # keep a small tail in case SOI crosses chunk boundary
#                 if len(self.buf) > 2048:
#                     del self.buf[:-2048]
#                 continue
#             e = self.buf.find(EOI, s + 2)
#             if e < 0:
#                 # keep from SOI onward
#                 if s > 0:
#                     del self.buf[:s]
#                 continue
#             # slice complete JPEG
#             e2 = e + 2
#             jpg = bytes(self.buf[s:e2])
#             del self.buf[:e2]
#             return jpg

#     async def stop(self) -> None:
#         try:
#             if self.proc and self.proc.returncode is None:
#                 self.proc.terminate()
#         except Exception:
#             pass

# ASCII only
import asyncio
from collections import deque
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
        self._buf = bytearray()
        self._stderr_tail = deque(maxlen=200)
        self._stderr_task: Optional[asyncio.Task] = None

    def _build_cmd(self) -> List[str]:
        return [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",  # keep useful warnings
            "-f", "dshow",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i", f"video={self.device_name}",
            "-vsync", "drop",               # your key fix: drop instead of queuing
            "-f", "mjpeg",
            "pipe:1",
        ]

    async def start(self) -> None:
        cmd = self._build_cmd()
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,  # capture for diagnostics
        )
        self._buf.clear()
        # background task to keep stderr drained (prevents blocking) and store last lines
        self._stderr_task = asyncio.create_task(self._pump_stderr())

    async def _pump_stderr(self):
        assert self.proc and self.proc.stderr
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    break
                # Store decoded tail, strip \n
                self._stderr_tail.append(line.decode(errors="replace").rstrip())
        except Exception:
            # don't crash the app if the pump hits an error
            pass

    def get_stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def is_running(self) -> bool:
        return bool(self.proc) and (self.proc.returncode is None)

    async def restart(self) -> None:
        await self.stop()
        await asyncio.sleep(0.1)
        await self.start()

    async def read_jpeg(self) -> Optional[bytes]:
        if not self.proc or not self.proc.stdout:
            return None
        read = self.proc.stdout.read
        while True:
            chunk = await read(65536)  # bigger read
            if not chunk:
                # EOF: ffmpeg ended or pipe broken
                return None
            self._buf.extend(chunk)

            s = self._buf.find(SOI)
            if s < 0:
                # keep small tail in case SOI straddles chunks
                if len(self._buf) > 2048:
                    del self._buf[:-2048]
                continue

            e = self._buf.find(EOI, s + 2)
            if e < 0:
                # keep from SOI onward
                if s > 0:
                    del self._buf[:s]
                continue

            e2 = e + 2
            jpg = bytes(self._buf[s:e2])
            del self._buf[:e2]
            return jpg

    async def stop(self) -> None:
        # stop stderr pump first
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._stderr_task = None

        try:
            if self.proc and self.proc.returncode is None:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=1.5)
                except asyncio.TimeoutError:
                    self.proc.kill()
        except Exception:
            pass
        finally:
            self.proc = None
