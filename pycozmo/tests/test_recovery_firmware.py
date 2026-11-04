import unittest
from unittest import mock

import pycozmo
from pycozmo.frame import Frame


# The first frame a robot on its recovery firmware sent, on 4 November 2026: events, debug lines, FWVersionInfo
# (0xd2) and RobotAvailable (0xc9). It sends no FirmwareSignature.
RECOVERY_FRAME = bytes.fromhex(
    "434f5a0352450109020005000100051100f1f01d830300000000000000000004ff00051200f423070000e2efb03908ab0fba36dc4e"
    "3a1e0b110083d83e89ed5fda40000000000000000000051100f12893830300000000000000000004ff00040d00b00200000001000301"
    "02000000042d00d266b1570866b1570800000000940fe597091708b4e16db3ca99ddde3fae00fcd5d1983a37f30d8943dcd8b5d10407"
    "00c9b76270090000040900b0bf01000028000200")


class TestFirmwareVersionInfo(unittest.TestCase):

    def test_decode(self):
        f = Frame.from_bytes(RECOVERY_FRAME)
        infos = [p for p in f.pkts if isinstance(p, pycozmo.protocol_encoder.FirmwareVersionInfo)]
        self.assertEqual(len(infos), 1)
        info = infos[0]
        self.assertEqual(info.wifi_version, 0x0857b166)
        self.assertEqual(info.rtip_version, 0x0857b166)
        self.assertEqual(info.body_version, 0)
        self.assertEqual(bytes(info.to_robot_clad_hash), bytes.fromhex("940fe597091708b4e16db3ca99ddde3f"))
        self.assertEqual(bytes(info.to_engine_clad_hash), bytes.fromhex("ae00fcd5d1983a37f30d8943dcd8b5d1"))

    def test_encode_decode(self):
        pkt = pycozmo.protocol_encoder.FirmwareVersionInfo(
            wifi_version=1, rtip_version=2, body_version=3, to_robot_clad_hash=tuple(range(16)),
            to_engine_clad_hash=tuple(range(16, 32)))
        self.assertEqual(len(pkt.to_bytes()), 44)
        decoded = pycozmo.protocol_encoder.FirmwareVersionInfo.from_bytes(pkt.to_bytes())
        self.assertEqual(decoded.body_version, 3)
        self.assertEqual(tuple(decoded.to_engine_clad_hash), tuple(range(16, 32)))


class TestRecoveryConnection(unittest.TestCase):
    """ A robot that answers on its recovery firmware is named as such, not taken for a network fault. """

    def setUp(self):
        # A client that is never started, with the connection stubbed out.
        self.cli = pycozmo.client.Client()
        patcher = mock.patch.object(self.cli.conn, "send")
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_a_robot_in_recovery_is_named(self):
        info = pycozmo.protocol_encoder.FirmwareVersionInfo(
            wifi_version=0x0857b166, rtip_version=0x0857b166, to_robot_clad_hash=(0,) * 16,
            to_engine_clad_hash=(0,) * 16)
        self.cli._on_firmware_version_info(self.cli.conn, info)
        self.assertIs(self.cli.firmware_version_info, info)
        with self.assertRaises(pycozmo.exception.ConnectionTimeout) as cm:
            self.cli.wait_for_robot(timeout=0.01)
        self.assertIn("recovery firmware", str(cm.exception))
        self.assertIn("body 0x00000000", str(cm.exception))

    def test_a_silent_robot_is_not(self):
        with self.assertRaises(pycozmo.exception.ConnectionTimeout) as cm:
            self.cli.wait_for_robot(timeout=0.01)
        self.assertEqual(str(cm.exception), "Failed to connect to Cozmo.")

    def test_a_signature_needs_no_version_info(self):
        sig = pycozmo.protocol_encoder.FirmwareSignature(signature='{"version": 2381, "build": "DEVELOPMENT"}')
        self.cli._on_firmware_signature(self.cli.conn, sig)
        assert self.cli.robot_fw_sig is not None
        self.assertEqual(self.cli.robot_fw_sig["version"], 2381)
        self.assertIsNone(self.cli.firmware_version_info)
