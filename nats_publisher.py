from typing import Iterable, Optional
import json
import numpy as np
import asyncio
import nats

class NatsPublisher:
    def __init__(self, servers: Iterable[str], subject: str, enable: bool = True):
        self.servers = list(servers)
        self.subject = subject
        self.enable = enable
        self.conns = []

    async def connect(self):
        if not self.enable:
            return
        for url in self.servers:
            nc = await nats.connect(url)
            self.conns.append(nc)
        if self.conns:
            print("Connected to NATS:", ", ".join(self.servers))

    async def close(self):
        if not self.conns:
            return
        for nc in self.conns:
            try:
                await nc.flush(timeout=2)
            except Exception:
                pass
            try:
                await nc.close()
            except Exception:
                pass
        self.conns.clear()

    @staticmethod
    def _to_json_bytes(obj) -> bytes:
        def _default(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Type {type(o).__name__} not JSON serializable")
        return json.dumps(obj, default=_default, separators=(",", ":")).encode("utf-8")

    async def publish_xy(self, x_mm: float, y_mm: float, angle_deg: float = 0.0):
        if not self.enable or not self.conns:
            return
        payload = self._to_json_bytes({"x": x_mm, "y": y_mm, "angle": angle_deg})
        for nc in self.conns:
            await nc.publish(self.subject, payload)