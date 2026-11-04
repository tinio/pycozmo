#!/usr/bin/env python
"""

Cozmo robot over-the-air (OTA) firmware update application.

"""

from typing import BinaryIO, Optional
import sys
import os
import json
import math
from threading import Event
import time
import argparse

import pycozmo


safe_file = ""
verbose = False

chunk_id = 0
evt = Event()

last_chunk_id = -1
last_status = -1


class UpdateError(Exception):
    """ Update exception. """
    pass


def on_firmware_update_result(_: pycozmo.conn.Connection,
                              pkt: pycozmo.protocol_encoder.FirmwareUpdateResult) -> None:
    """ FirmwareUpdateResult packet handler. """

    global last_chunk_id
    global last_status

    last_chunk_id = pkt.chunk_id
    last_status = pkt.status

    if verbose:
        print("Progress: {} bytes / {} chunks; status {}".format(pkt.byte_count, last_chunk_id, last_status))

    evt.set()


def wait_for_result(timeout: Optional[float] = None) -> None:
    """ Wait for result message. """

    evt.wait(timeout=timeout)
    evt.clear()

    # 0 indicates operation success and 10 indicates update completion.
    if last_status and last_status != 10:
        raise UpdateError("Update failed with error code {}.".format(last_status))


def send_chunk(cli: pycozmo.client.Client, f: BinaryIO) -> bool:
    """ Read and send a chunk. """

    global chunk_id

    chunk = f.read(1024)
    if chunk:
        if len(chunk) < 1024:
            # Pad last chunk.
            chunk = chunk.ljust(1024, b"\0")
            res = True
        else:
            res = False

        pkt = pycozmo.protocol_encoder.FirmwareUpdate(chunk_id=chunk_id, data=chunk)
        cli.conn.send(pkt)

        chunk_id += 1
    else:
        # Reached end of file.
        res = True

    return res


def update(cli: pycozmo.client.Client) -> None:
    """ Perform robot OTA firmware update. """

    # Register for FirmwareUpdateResult packets.
    cli.add_handler(pycozmo.protocol_encoder.FirmwareUpdateResult, on_firmware_update_result)

    safe_size = os.path.getsize(safe_file)
    total_chunks = math.ceil(safe_size / 1024)
    if verbose:
        print("Safe size: {} bytes / {} chunks".format(safe_size, total_chunks))

    with open(safe_file, "rb") as f:

        print("Initiating update...")
        send_chunk(cli, f)
        wait_for_result(30.0)
        if last_status != 0:
            raise UpdateError("Failed to receive initialization confirmation.")

        print("Transferring firmware...")
        done = False
        while not done:
            if verbose:
                print("Sending chunk {}/{}...".format(chunk_id+1, total_chunks))
            # For some reason the robot sends FirmwareUpdateResult only every 2 chunks.
            send_chunk(cli, f)
            done = send_chunk(cli, f)
            wait_for_result(0.5)

    # Finalize update
    print("Finalizing update...")
    pkt = pycozmo.protocol_encoder.FirmwareUpdate(chunk_id=0xFFFF, data=b"\0" * 1024)
    cli.conn.send(pkt)
    # Factory FW may first ack with status 0 then later 10 after flashing.
    # Poll for up to 30s for status 10.
    try:
        wait_for_result(10.0)
    except UpdateError:
        pass
    if last_status != 10:
        # Retry waiting: robot may need more time to flash and reboot.
        for _ in range(6):
            time.sleep(5)
            try:
                wait_for_result(0.2)
            except UpdateError:
                pass
            if last_status == 10:
                break
            # Re-send termination if stuck at 0 or 1
            if last_status in (0, 1):
                cli.conn.send(pkt)
                try:
                    wait_for_result(5.0)
                    if last_status == 10:
                        break
                except UpdateError:
                    pass
    if last_status != 10:
        raise UpdateError("Failed to receive update confirmation (status {}).".format(last_status))

    time.sleep(15.0)

    print("Done.")


def verify_safe(fspec: str, signature: bool) -> None:
    """ Perform .safe file sanity check. """

    try:
        with open(fspec, "rb") as f:
            raw_sig = f.read(1024).decode("utf-8").rstrip("\0")
        sig = json.loads(raw_sig)
    except Exception as e:
        print("ERROR: Failed to read signature from .safe file. {}".format(e))
        sys.exit(1)

    for k in ("version", "wifiSig", "rtipSig", "bodySig"):
        if k not in sig:
            print("ERROR: Invalid .safe file.")
            sys.exit(2)

    if signature:
        print(json.dumps(sig, indent=4, separators=(",", ": ")))
        sys.exit(0)
    else:
        print("Updating to version {}...".format(sig["version"]))


def parse_args():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("-s", "--signature", action="store_true", help="Dump .safe signature and exit")
    parser.add_argument("safe_file", help=".safe file specification")
    args = parser.parse_args()
    return args


def main():
    global safe_file
    global verbose

    # Parse command-line.
    args = parse_args()

    safe_file = args.safe_file
    verbose = args.verbose

    # Verify .safe file.
    verify_safe(safe_file, args.signature)

    # Update robot.
    try:
        with pycozmo.connect(protocol_log_level="INFO", robot_log_level="DEBUG", auto_initialize=False) as cli:
            update(cli)
    except Exception as e:
        print("ERROR: {}".format(e))
        sys.exit(3)


if __name__ == '__main__':
    main()
