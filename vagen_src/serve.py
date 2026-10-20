"""
SRC environment server for VAGEN.

Run this on the machine where AMBF + ROS + surgical_robotics_challenge run,
after the simulator is up. VAGEN training / evaluation then connects to it with
the generic `RemoteEnv` client (see configs/*.yaml).

Usage:
    python -m vagen_src.serve --task approach --port 8000
    python -m vagen_src.serve --task place --port 8001 --camera_topic ""   # text-only
"""

from __future__ import annotations

import argparse
import logging

from .paths import setup_sim_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="approach", help="approach | place | insert | regrasp | pullout")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trans_error", type=float, default=0.5, help="success threshold, cm (SurgicAI convention)")
    parser.add_argument("--angle_error", type=float, default=30.0, help="success threshold, degrees")
    parser.add_argument("--max_low_level_steps", type=int, default=300)
    parser.add_argument("--camera_topic", default="/ambf/env/cameras/cameraL/ImageData",
                        help='ROS image topic; "" for text-only observations')
    parser.add_argument("--image_size", type=int, nargs=2, default=[448, 336], metavar=("W", "H"))
    parser.add_argument("--flip_image", action="store_true")
    parser.add_argument("--session_timeout", type=float, default=600.0)
    parser.add_argument("--api_key", default="")
    args = parser.parse_args()

    setup_sim_paths()

    import uvicorn
    from vagen.envs_remote import GymService

    from .handler import SRCHandler
    from .sim import SRCSim

    # Must be created on the main thread (AMBF client -> rospy.init_node).
    sim = SRCSim(
        task=args.task,
        trans_error=args.trans_error,
        angle_error=args.angle_error,
        max_low_level_steps=args.max_low_level_steps,
        camera_topic=args.camera_topic or None,
        image_size=tuple(args.image_size),
        flip_image=args.flip_image,
    )
    handler = SRCHandler(sim, session_timeout=args.session_timeout)
    app = GymService(handler, api_key=args.api_key).build()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
