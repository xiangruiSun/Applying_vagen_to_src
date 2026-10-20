"""
Grab RGB frames from the AMBF stereo cameras over ROS 1.

AMBF publishes `sensor_msgs/Image` on /ambf/env/cameras/<name>/ImageData when
the world file has `publish image: True` (SRC's ADF/world/world_stereo.yaml
does this for cameraL / cameraR). The ROS node itself is initialised by the
AMBF client that SurgicAI's env creates, so we only subscribe here.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image


class RosCamera:
    def __init__(
        self,
        topic: str = "/ambf/env/cameras/cameraL/ImageData",
        resize: Optional[Tuple[int, int]] = (448, 336),
        flip_vertical: bool = False,
    ):
        import rospy
        from sensor_msgs.msg import Image as ImageMsg

        self.topic = topic
        self.resize = tuple(resize) if resize else None
        self.flip_vertical = flip_vertical
        self._lock = threading.Lock()
        self._latest = None
        self._stamp = 0.0
        self._sub = rospy.Subscriber(topic, ImageMsg, self._cb, queue_size=1, buff_size=2 ** 24)

    def _cb(self, msg) -> None:
        with self._lock:
            self._latest = msg
            self._stamp = time.time()

    def grab(self, timeout: float = 5.0, newer_than: float = 0.0) -> Image.Image:
        """Return the latest frame received after `newer_than` (time.time())."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                msg, stamp = self._latest, self._stamp
            if msg is not None and stamp > newer_than:
                return self._to_pil(msg)
            if time.time() > deadline:
                raise TimeoutError(
                    f"No image on {self.topic} within {timeout}s. Is the simulator running with "
                    f"`publish image: True` for this camera?"
                )
            time.sleep(0.01)

    def _to_pil(self, msg) -> Image.Image:
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
        if channels is None:
            raise ValueError(f"Unsupported image encoding {msg.encoding!r} on {self.topic}")
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step // channels, channels)
        arr = arr[:, : msg.width, :3]
        if msg.encoding.startswith("bgr"):
            arr = arr[:, :, ::-1]
        if self.flip_vertical:
            arr = arr[::-1]
        img = Image.fromarray(np.ascontiguousarray(arr), mode="RGB")
        if self.resize:
            img = img.resize(self.resize, Image.BILINEAR)
        return img
