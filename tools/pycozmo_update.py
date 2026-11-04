#!/usr/bin/env python
"""

Cozmo robot over-the-air (OTA) firmware update application.

"""

from typing import BinaryIO, Dict, Optional
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
# None until the robot first answers: "no answer" used to read as -1, which is also ERR_NO_MEM.
last_status: Optional[int] = None

#: The robot's OTA results, as Anki's engine names them (libcozmoEngine.so of the 3.4.0 application). The status
#: byte is signed.
OKAY = 0
SUCCESS = 10
OTA_RESULTS: Dict[int, str] = {
    0: "OKAY", 1: "BUSY", 2: "NOTHING_TO_DO", 3: "OUT_OF_ORDER", 4: "NOT_ENABLED", 5: "NOT_ON_CHARGER",
    10: "SUCCESS",
    -1: "ERR_NO_MEM", -2: "ERR_BAD_ADDRESS", -3: "ERR_BAD_DATA", -4: "ERR_BAD_ENC", -5: "ERR_I2SPI",
    -6: "ERR_RTIP_NAK", -7: "ERR_RTIP_TIMEOUT", -8: "ERR_INCOMPATIBLE", -10: "ERR_FACT_RESET",
    -20: "ERR_ERASE_TIMEOUT", -21: "ERR_ERASE_ERROR", -22: "ERR_WRITE_TIMEOUT", -23: "ERR_WRITE_ERROR",
    -24: "ERR_READ_TIMEOUT", -25: "ERR_READ_ERROR",
}


def describe(status: Optional[int]) -> str:
    """ Name an OTA result. """
    if status is None:
        return "no answer"
    return "{} ({})".format(OTA_RESULTS.get(status, "unknown"), status)


class UpdateError(Exception):
    """ Update exception. """
    pass


def on_firmware_update_result(_: pycozmo.conn.Connection,
                              pkt: pycozmo.protocol_encoder.FirmwareUpdateResult) -> None:
    """ FirmwareUpdateResult packet handler. """

    global last_chunk_id
    global last_status

    last_chunk_id = pkt.chunk_id
    last_status = pkt.status - 256 if pkt.status > 127 else pkt.status

    if verbose:
        print("Progress: {} bytes / {} chunks; {}".format(pkt.byte_count, last_chunk_id, describe(last_status)))

    evt.set()


def wait_for_result(timeout: Optional[float] = None) -> bool:
    """ Wait for a result message. Raise on an error; return whether a result came. """

    answered = evt.wait(timeout=timeout)
    evt.clear()

    # OKAY acknowledges a chunk and SUCCESS the whole update.
    if last_status is not None and last_status not in (OKAY, SUCCESS):
        raise UpdateError("The robot reported {}.".format(describe(last_status)))
    return answered


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

    global last_status

    # Register for FirmwareUpdateResult packets.
    cli.add_handler(pycozmo.protocol_encoder.FirmwareUpdateResult, on_firmware_update_result)

    safe_size = os.path.getsize(safe_file)
    total_chunks = math.ceil(safe_size / 1024)
    if verbose:
        print("Safe size: {} bytes / {} chunks".format(safe_size, total_chunks))

    with open(safe_file, "rb") as f:

        print("Initiating update...")
        send_chunk(cli, f)
        # The robot erases its flash before it answers, which took 8 s on one.
        if not wait_for_result(30.0) or last_status != OKAY:
            raise UpdateError("The robot did not accept the first chunk: {}.".format(describe(last_status)))

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
    # The last chunk's OKAY is not an answer to the end marker.
    last_status = None
    pkt = pycozmo.protocol_encoder.FirmwareUpdate(chunk_id=0xFFFF, data=b"\0" * 1024)
    cli.conn.send(pkt)
    wait_for_result(10.0)
    if last_status != SUCCESS:
        raise UpdateError("The robot did not confirm the update: {}.".format(describe(last_status)))

    time.sleep(15.0)

    print("Done.")


def connect() -> pycozmo.client.Client:
    """ Connect to the robot, on its application firmware or on its recovery firmware. """

    pycozmo.setup_basic_logging(protocol_log_level="INFO", robot_log_level="DEBUG")
    cli = pycozmo.client.Client(auto_initialize=False)
    cli.start()
    cli.connect()
    try:
        cli.wait_for_robot()
    except pycozmo.exception.ConnectionTimeout:
        if cli.firmware_version_info is None:
            cli.stop()
            raise
        # A robot on its recovery firmware sends no firmware signature, the cue on which the client asks for its
        # manufacturing info. Every update that went through on such a robot asked for it first.
        print("The robot is running its recovery firmware.")
        cli._enable_robot()
        time.sleep(1.0)
    return cli


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
        cli = connect()
    except Exception as e:
        print("ERROR: {}".format(e))
        sys.exit(3)
    try:
        update(cli)
    except Exception as e:
        print("ERROR: {}".format(e))
        sys.exit(3)
    finally:
        cli.disconnect()
        cli.stop()


if __name__ == '__main__':
    main()
