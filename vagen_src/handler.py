"""
VAGEN remote-env handler for SRC.

One AMBF simulator can only run one episode at a time, so the handler hands
out a single slot: a new session waits until the previous one is closed (or
times out). To run rollouts in parallel, launch several simulators, each with
its own ROS master and server port, and list all URLs in `base_urls`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from vagen.envs_remote import BaseGymHandler

from .sim import SRCSim
from .src_env import SRCEnv

LOGGER = logging.getLogger(__name__)


class SRCHandler(BaseGymHandler):
    def __init__(self, sim: SRCSim, session_timeout: float = 600.0, slot_wait_timeout: float = 600.0):
        super().__init__(session_timeout=session_timeout, max_sessions=0)
        self.sim = sim
        self.slot_wait_timeout = slot_wait_timeout
        # Created lazily: on Python 3.8 asyncio primitives bind to the loop
        # that exists at construction time, which is not uvicorn's loop.
        self._slot: Optional[asyncio.Semaphore] = None

    async def create_env(self, env_config: Dict[str, Any]) -> SRCEnv:
        if self._slot is None:
            self._slot = asyncio.Semaphore(1)
        try:
            await asyncio.wait_for(self._slot.acquire(), timeout=self.slot_wait_timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Simulator busy for {self.slot_wait_timeout}s; retry later.")
        try:
            return SRCEnv(env_config, sim=self.sim, on_close=self._slot.release)
        except Exception:
            self._slot.release()
            raise
