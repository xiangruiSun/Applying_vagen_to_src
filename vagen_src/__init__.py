"""
vagen_src: glue code that exposes the Surgical Robotics Challenge (SRC)
subtasks defined by SurgicAI as a VAGEN-lite multi-turn VLM environment.

Layout:
    paths.py    - puts the third_party submodules on sys.path
    tasks.py    - SurgicAI subtask registry + natural-language instructions
    parse.py    - VLM response / mid-level primitive parsing
    prompt.py   - system prompt and observation templates
    camera.py   - ROS camera image grabber
    sim.py      - SRCSim: owns one SurgicAI env and executes primitives
    src_env.py  - SRCEnv: GymImageEnv implementation used by VAGEN
    handler.py  - VAGEN remote-env handler (one session per simulator)
    serve.py    - HTTP server entry point (runs next to AMBF/ROS)
"""
