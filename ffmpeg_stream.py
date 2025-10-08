# ASCII only
import asyncio
import contextlib
from collections import deque
from typing import List, Optional

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"

class FFMPEGMJPEGStream:
    """
    ffmpeg (dshow) -> MJPEG frames to stdout -> parse SOI/EOI.
    Diagnostics-heavy: prints exact ffmpeg command, PID, stderr tail, counters,
    and buffer state whenever EOF or anomalies occur.
    """

    def __init__(self, device_name: str, width: int, height: int, fps: int,
                 loglevel: str = "info"):
        self.device_name = device_name
        self.width = width
        self.height = height
        self.fps = fps
        self.fflog = loglevel  # "info", "verbose", or "debug"

        self.proc: Optional[asyncio.subprocess.Process] = None
        self._buf = bytearray()

        # stderr diagnostics
        self._stderr_tail = deque(maxlen=200)
        self._stderr_task: Optional[asyncio.Task] = None

        # counters
        self._frames = 0
        self._bytes = 0
        self._reads_without_frame = 0  # watchdog for stuck parsing

    def _build_cmd(self) -> List[str]:
        return [
            "ffmpeg",
            "-hide_banner", "-loglevel", "info",

            # Input: DirectShow camera, ask for MJPEG from the device
            "-f", "dshow",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-vcodec", "mjpeg",
            "-i", f"video={self.device_name}",

            # Make sure timestamps move forward & drop if consumer is slower
            "-fflags", "+genpts",
            "-use_wallclock_as_timestamps", "1",
            "-fps_mode", "drop",          # modern replacement for -vsync drop

            # <<< KEY CHANGE: do NOT encode; copy the MJPEG bitstream >>>
            "-c:v", "copy",

            # Output: MJPEG stream to stdout
            "-f", "mjpeg",
            "pipe:1",
        ]


    async def start(self) -> None:
        cmd = self._build_cmd()
        print("[FFMPEG] starting:", " ".join(cmd))
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,   # capture for diagnostics
        )
        print(f"[FFMPEG] pid={self.proc.pid if self.proc else 'n/a'}")
        self._buf.clear()
        self._frames = 0
        self._bytes = 0
        self._reads_without_frame = 0
        # drain stderr in the background
        self._stderr_task = asyncio.create_task(self._pump_stderr())

    async def _pump_stderr(self):
        assert self.proc and self.proc.stderr
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    break  # EOF -> exit cleanly
                s = line.decode(errors="replace").rstrip()
                self._stderr_tail.append(s)
                if ("Error" in s) or ("error" in s) or ("fail" in s) or ("Invalid" in s):
                    print("[FFMPEG][stderr]", s)
        except Exception as e:
            print(f"[FFMPEG][stderr pump] exception: {e!r}")

    def get_stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def is_running(self) -> bool:
        return bool(self.proc) and (self.proc.returncode is None)

    def _dump_state(self, why: str) -> None:
        rc = self.proc.returncode if self.proc else None
        pid = self.proc.pid if self.proc else None
        print("\n========== FFMPEG STREAM DIAGNOSTIC DUMP ==========")
        print(f"[why]             {why}")
        print(f"[running]         {self.is_running()}  (pid={pid}, returncode={rc})")
        print(f"[buf_len]         {len(self._buf)} bytes")
        print(f"[frames_parsed]   {self._frames}")
        print(f"[bytes_read]      {self._bytes}")
        # show a small hexdump of tail to see if we are mid-JPEG
        tail = bytes(self._buf[-64:]) if len(self._buf) else b""
        print(f"[buf_tail_hex]    {tail.hex(' ')}")
        # stderr tail
        tail_txt = self.get_stderr_tail()
        print("---- ffmpeg stderr (last lines) ----")
        if tail_txt:
            print(tail_txt)
        else:
            print("(no stderr yet)")
        print("===================================================\n")

    async def read_jpeg(self) -> Optional[bytes]:
        if not self.proc or not self.proc.stdout:
            print("[FFMPEG] read_jpeg called but process/stdout not ready")
            self._dump_state("proc/stdout not ready")
            return None

        read = self.proc.stdout.read

        while True:
            try:
                chunk = await read(65536)  # 64 KiB reads reduce syscall overhead
            except Exception as e:
                print(f"[FFMPEG] stdout.read exception: {e!r}")
                self._dump_state("stdout.read exception")
                return None

            if not chunk:
                # EOF: process ended or pipe closed. This is the ONLY reason read() yields b''.
                print("[FFMPEG] EOF on stdout (no bytes)")
                self._dump_state("EOF on stdout")
                return None

            self._bytes += len(chunk)
            self._buf.extend(chunk)

            # Try to extract the latest complete JPEG
            s = self._buf.find(SOI)
            if s < 0:
                # keep a tiny tail in case marker straddles boundary
                if len(self._buf) > 2048:
                    del self._buf[:-2048]
                self._reads_without_frame += 1
            else:
                e = self._buf.find(EOI, s + 2)
                if e < 0:
                    if s > 0:
                        del self._buf[:s]
                    self._reads_without_frame += 1
                else:
                    e2 = e + 2
                    jpg = bytes(self._buf[s:e2])
                    del self._buf[:e2]
                    self._frames += 1
                    self._reads_without_frame = 0
                    return jpg

            # Watchdog: if we keep reading but never complete a JPEG, tell us
            if self._reads_without_frame in (50, 200, 1000):
                self._dump_state(f"watchdog: {self._reads_without_frame} reads, no full JPEG yet")

    async def stop(self) -> None:
        # 1) Terminate the child and wait for it to exit
        try:
            if self.proc and self.proc.returncode is None:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.proc.kill()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self.proc.wait(), timeout=2.0)
        except Exception as e:
            print(f"[FFMPEG] stop() terminate/kill exception: {e!r}")

        # 2) Let the stderr pump finish on its own (EOF), then await it
        if self._stderr_task:
            try:
                # Give it a moment to read EOF and exit
                await asyncio.wait_for(self._stderr_task, timeout=1.5)
            except asyncio.TimeoutError:
                # If it didn't finish, cancel politely and await
                self._stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._stderr_task
            finally:
                self._stderr_task = None

        # 3) Explicitly close underlying transports to avoid Proactor repr touching closed fds
        try:
            if self.proc and self.proc.stdout and getattr(self.proc.stdout, "_transport", None):
                self.proc.stdout._transport.close()
        except Exception:
            pass
        try:
            if self.proc and self.proc.stderr and getattr(self.proc.stderr, "_transport", None):
                self.proc.stderr._transport.close()
        except Exception:
            pass

        # 4) Drop the proc reference
        self.proc = None