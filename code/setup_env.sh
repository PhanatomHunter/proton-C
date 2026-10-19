#!/bin/sh
# Installed packages do not persist between conversation turns in this sandbox.
# Run this before any script that needs ffmpeg (bench.py, cli.py bench mode).
pip3 install -q imageio-ffmpeg >/dev/null 2>&1
python3 -c "import imageio_ffmpeg as m; print('ffmpeg ready:', m.get_ffmpeg_exe())"
