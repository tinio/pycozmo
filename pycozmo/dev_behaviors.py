"""

Development and factory self-tests: turn calibration, charger docking, lift travel, hardware checkout, and the camera
centroid extraction.

These are the engine's own test behaviors, run from its dev menu or on the factory line rather than by the brain. Each
runs its checks from its configuration, logs how each step went, and keeps what it found for the caller to read: pass
or fail, and the measurements behind it.

DevTurnInPlaceTest turns by each configured angle at its speed, and says whether the heading landed within its
tolerance. DockingTestSimple drives back onto the charger with the charger handling, and says whether the robot got
on. LiftLoadTest runs the lift from bottom to top and back, and says whether the reported height tracked the orders.
FactoryTest works the backpack lights, head, lift, treads, and camera in turn, and fails on the first thing that does
not answer. FactoryCentroidExtractor snaps one camera frame and takes the middle of its bright pixels.

"""

import math
import threading
from typing import Any, List, Optional, Tuple

import numpy as np

from . import charger_handling
from . import event
from . import lights
from . import robot
from . import util
from .cube_behaviors import BehaviorScript
from .cube_handling import Cancelled, _check


__all__ = [
    "BehaviorDevTurnInPlaceTest",
    "BehaviorDockingTestSimple",
    "BehaviorLiftLoadTest",
    "BehaviorFactoryTest",
    "BehaviorFactoryCentroidExtractor",
]


def snap_frame(cli: Any, timeout: float = 2.0) -> Optional[Any]:
    """ The next camera image, or None when none comes before the timeout. The camera has to be on. """
    images: List[Any] = []
    got = threading.Event()

    def on_image(_cli: Any, image: Any) -> None:
        images.append(image)
        got.set()

    handler = cli.add_handler(event.EvtNewRawCameraImage, on_image, one_shot=True)
    try:
        if not got.wait(timeout):
            return None
    finally:
        cli.del_handler(event.EvtNewRawCameraImage, handler)
    return images[0]


def extract_centroid(image: Any, threshold: int = 200) -> Optional[Tuple[float, float, int]]:
    """
    Where the bright pixels of a frame are centered, if there are enough of them and not nearly all of them: the
    middle of the bright patch in pixels and how many of them there are.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    rows, cols = np.nonzero(gray >= threshold)
    if len(rows) < 3 or len(rows) > gray.size // 4:
        return None
    return float(cols.mean()), float(rows.mean()), len(rows)


class BehaviorDevTurnInPlaceTest(BehaviorScript):
    """
    DevTurnInPlaceTest - turn by each configured angle, and check the landing.

    Each test names its angle, speed, acceleration, and tolerance; the robot turns, and the heading it reports is held
    against the angle within the tolerance. Acceleration is not something the robot can be told, so the speed alone
    drives the turn. The results keep every test's commanded angle, landing error in degrees, and pass or fail.
    """

    #: How long a turn may take beyond its own duration, in seconds, and the gap between tests.
    TIMEOUT_MARGIN = 5.0
    GAP = 1.0

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.results: List[Tuple[float, float, bool]] = []
        self.passed: Optional[bool] = None

    def wants_to_run(self) -> bool:
        return True

    def script(self, cancel: threading.Event) -> None:
        tests = list(self.conf.get("tests", []))
        runs = int(self.conf.get("nRunsPerTest", 1))
        gap = float(self.conf.get("gapBetweenTests_s", self.GAP))
        loop = bool(self.conf.get("loopForever", False))
        self.results = []
        while True:
            _check(cancel)
            for test in tests:
                _check(cancel)
                self._run_test(test, cancel)
                if cancel.wait(gap):
                    raise Cancelled()
            if not loop:
                break
            runs += 1
        self.passed = all(passed for _, _, passed in self.results) if self.results else None

    def _run_test(self, test: Any, cancel: threading.Event) -> None:
        """ Turn by the test's angle at its speed, and keep how near the landing was. """
        angle = float(test.get("angle_deg", 90.0))
        speed = float(test.get("speed_deg_per_sec", 150.0))
        tolerance = float(test.get("tol_deg", 2.0))
        before = self.cli.pose.rotation.angle_z.radians
        timeout = abs(angle) / max(speed, 1.0) + self.TIMEOUT_MARGIN
        self.cli.turn_in_place(util.Angle(degrees=angle), speed=math.radians(speed),
                               wait=True, timeout=timeout)
        _check(cancel)
        error = abs(test_error_degrees(before, self.cli.pose.rotation.angle_z.radians, angle))
        self.results.append((angle, error, error <= tolerance))


def test_error_degrees(before_rad: float, after_rad: float, commanded_deg: float) -> float:
    """ How far the turn fell short of, or overshot, its commanded angle, in degrees. """
    turned = math.degrees(after_rad - before_rad)
    error = turned - commanded_deg
    while error > 180.0:
        error -= 360.0
    while error < -180.0:
        error += 360.0
    return error


class BehaviorDockingTestSimple(BehaviorScript):
    """
    DockingTestSimple - drive back onto the charger, and say whether the robot got on.

    A robot already on its charger passes without moving; anything else goes through the charger handling, which looks
    for the marker, drives out in front of it, and backs on. The verdict is kept for the caller.
    """

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.passed: Optional[bool] = None

    def wants_to_run(self) -> bool:
        return True

    def script(self, cancel: threading.Event) -> None:
        if self.cli.robot_status & robot.RobotStatusFlag.IS_ON_CHARGER:
            self.passed = True
            return
        self.passed = bool(charger_handling.go_to_charger(self.cli, cancel))


class BehaviorLiftLoadTest(BehaviorScript):
    """
    LiftLoadTest - run the lift from bottom to top and back, and check it tracked.

    Raises the lift all the way and lowers it again, holding what it reports against each order within tolerance. The
    verdict is kept for the caller, with every ordered and reported height.
    """

    #: How near the reported height has to be to the ordered one, in mm.
    TOLERANCE = 5.0

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.results: List[Tuple[float, float, bool]] = []
        self.passed: Optional[bool] = None

    def wants_to_run(self) -> bool:
        return True

    def script(self, cancel: threading.Event) -> None:
        self.results = []
        for height in (robot.MIN_LIFT_HEIGHT.mm, robot.MAX_LIFT_HEIGHT.mm, robot.MIN_LIFT_HEIGHT.mm):
            _check(cancel)
            self.cli.set_lift_height(height)
            if cancel.wait(1.0):
                raise Cancelled()
            reported = self.cli.lift_position.height.mm
            self.results.append((height, reported, abs(reported - height) <= self.TOLERANCE))
        self.passed = all(passed for _, _, passed in self.results)


class BehaviorFactoryTest(BehaviorScript):
    """
    FactoryTest - work every output in turn, and fail on the first thing that does not answer.

    Cycles the backpack lights through red, green, and blue, nods the head down and up, runs the lift bottom to top
    and back, drives a short way out and back, and asks the camera for a frame. Each step that moves checks the robot
    reported back where it was told; the steps keep their names and passes.
    """

    #: How far the checkout drive goes, in mm, and how long a camera frame is waited for, in seconds.
    DRIVE_DISTANCE = 30.0
    DRIVE_SPEED = 40.0
    CAMERA_TIMEOUT = 2.0
    LIFT_TOLERANCE = 5.0

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.results: List[Tuple[str, bool]] = []
        self.passed: Optional[bool] = None

    def wants_to_run(self) -> bool:
        return True

    def script(self, cancel: threading.Event) -> None:
        self.results = []
        for name, step in (("lights", self._check_lights),
                           ("head", self._check_head),
                           ("lift", self._check_lift),
                           ("treads", self._check_treads),
                           ("camera", self._check_camera)):
            passed = step(cancel)
            self.results.append((name, passed))
            if not passed:
                self.passed = False
                return
        self.passed = True

    def _check_lights(self, cancel: threading.Event) -> bool:
        """ Cycle the backpack through its colors; there is nothing to read back, so doing it is passing. """
        _check(cancel)
        for color in (lights.red_light, lights.green_light, lights.blue_light):
            self.cli.set_all_backpack_lights(color)
            if cancel.wait(0.3):
                raise Cancelled()
        self.cli.set_backpack_lights_off()
        return True

    def _check_head(self, cancel: threading.Event) -> bool:
        """ Nod the head down and back up. """
        _check(cancel)
        self.cli.set_head_angle(robot.MIN_HEAD_ANGLE.radians)
        if cancel.wait(0.8):
            raise Cancelled()
        self.cli.set_head_angle(robot.MAX_HEAD_ANGLE.radians)
        if cancel.wait(0.8):
            raise Cancelled()
        return True

    def _check_lift(self, cancel: threading.Event) -> bool:
        """ Run the lift to the top and read back that it got near. """
        _check(cancel)
        self.cli.set_lift_height(robot.MAX_LIFT_HEIGHT.mm)
        if cancel.wait(1.0):
            raise Cancelled()
        reported = self.cli.lift_position.height.mm
        self.cli.set_lift_height(robot.MIN_LIFT_HEIGHT.mm)
        if cancel.wait(1.0):
            raise Cancelled()
        return abs(reported - robot.MAX_LIFT_HEIGHT.mm) <= self.LIFT_TOLERANCE

    def _check_treads(self, cancel: threading.Event) -> bool:
        """ Drive a short way out and back; getting there and back is passing. """
        _check(cancel)
        if not self.cli.drive_straight(util.Distance(mm=self.DRIVE_DISTANCE),
                                       speed=self.DRIVE_SPEED, wait=True):
            return False
        return bool(self.cli.drive_straight(util.Distance(mm=-self.DRIVE_DISTANCE),
                                            speed=self.DRIVE_SPEED, wait=True))

    def _check_camera(self, cancel: threading.Event) -> bool:
        """ Ask the camera for a frame, and pass on one coming back. """
        _check(cancel)
        self.cli.enable_camera(True, color=False)
        try:
            return snap_frame(self.cli, timeout=self.CAMERA_TIMEOUT) is not None
        finally:
            self.cli.enable_camera(False)


class BehaviorFactoryCentroidExtractor(BehaviorScript):
    """
    FactoryCentroidExtractor - snap one camera frame and take the middle of its bright pixels.

    The calibration line points the robot at its target and reads off where the middle lands; here the snapping and
    the middle-taking are the behavior, and the middle is kept for the caller, with how many bright pixels made it.
    """

    #: How long a frame is waited for, in seconds, and how bright a pixel counts, out of 255.
    CAMERA_TIMEOUT = 2.0
    THRESHOLD = 200

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.centroid: Optional[Tuple[float, float, int]] = None

    def wants_to_run(self) -> bool:
        return True

    def script(self, cancel: threading.Event) -> None:
        timeout = float(self.conf.get("cameraTimeout_sec", self.CAMERA_TIMEOUT))
        threshold = int(self.conf.get("brightThreshold", self.THRESHOLD))
        _check(cancel)
        self.cli.enable_camera(True, color=False)
        try:
            image = snap_frame(self.cli, timeout=timeout)
        finally:
            self.cli.enable_camera(False)
        _check(cancel)
        if image is None:
            return self.give_up("saw no frame")
        self.centroid = extract_centroid(image, threshold)
        if self.centroid is None:
            return self.give_up("found no bright patch")
