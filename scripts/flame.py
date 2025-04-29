#!/usr/bin/env python3
"""
collect_flame_data.py

Record two separate CSVs of MLX90640 frames:
  - flame.csv    (when there is a flame)
  - noflame.csv  (when there is no flame)
"""

import re
import sys
import csv
import serial
import serial.tools.list_ports
import numpy as np
from collections import deque

# ──────── CONFIG ──────── #
ROWS, COLS      = 24, 32
WORDS_PER_FRAME = 834            # 768 pixels + housekeeping
SERIAL_TIMEOUT  = 0.2            # seconds
HEX_PATTERN     = re.compile(r"^0[xX][0-9A-Fa-f]{4}$")
# ──────────────────────── #

def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        sys.exit("❌ No serial ports found.")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device:>8} — {p.description}")
    return ports

def open_serial():
    ports = list_ports()
    idx   = int(input("Select port index: "))
    baud  = int(input("Baud rate       : "))
    return serial.Serial(ports[idx].device, baud, timeout=SERIAL_TIMEOUT)

def get_one_frame(ser, fifo):
    """
    Read lines until we’ve collected exactly WORDS_PER_FRAME hex tokens,
    then return a (ROWS, COLS) uint16 NumPy array of the first 768.
    """
    while True:
        line = ser.readline().decode("utf-8", errors="ignore")
        if not line:
            continue
        for tok in re.split(r"[,\s]+", line):
            if HEX_PATTERN.match(tok):
                fifo.append(int(tok, 16))
        if len(fifo) >= WORDS_PER_FRAME:
            raw = [fifo.popleft() for _ in range(WORDS_PER_FRAME)]
            arr = np.array(raw[:ROWS*COLS], dtype=np.uint16)
            return arr.reshape(ROWS, COLS)

def record_frames(ser, fifo, count, out_csv):
    """Record `count` frames and write them to `out_csv`."""
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # header
        writer.writerow([f"p{i}" for i in range(ROWS*COLS)])
        for i in range(1, count+1):
            frame = get_one_frame(ser, fifo)
            writer.writerow(frame.flatten().tolist())
            print(f"  ✔ Recorded {i}/{count} → {out_csv}")

def main():
    ser  = open_serial()
    fifo = deque()
    print(f"\n▶ Streaming from {ser.port} …\n")

    # 1) Flame data
    n_flame = int(input("How many ‘flame’ frames to record? "))
    input("Light the flame in view of the camera, then press Enter to start…")
    record_frames(ser, fifo, n_flame, "flame.csv")
    print("\n✅ Saved flame.csv\n")

    # 2) No‑flame data
    n_nf = int(input("How many ‘no flame’ frames to record? "))
    input("Remove/cover the flame so there is NO flame, then press Enter…")
    record_frames(ser, fifo, n_nf, "noflame.csv")
    print("\n✅ Saved noflame.csv\n")

    ser.close()
    print("✔ Serial port closed. Done!")

if __name__ == "__main__":
    main()
