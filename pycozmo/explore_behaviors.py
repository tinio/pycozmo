"""

Explore behaviors: looking round, thinking about beacons, bringing cubes to them,
visiting edges and possible markers.

A beacon is an area the robot has been, a disc of beaconRadius_mm round where it
stood: beacons are how the hiking behaviors remember places to keep cubes in.
Memory maps, vantage raycasts and quad borders belong to Anki's engine and are
not here, so the edge and marker visits below do the observable part - go, look,
come back - and say so where they stop short.

"""

import math
import random
import threading
import time
import weakref
from typing import Any, List, Optional, Tuple

from . import camera, cube_handling, event, util
from .cube_behaviors import BehaviorScript, usable_cubes
from .cube_handling import Cancelled, _check, _wrap


__all__ = [
    "BehaviorExploreLookAroundInPlace",
    "BehaviorLookInPlaceMemoryMap",
    "BehaviorThinkAboutBeacons",
    "BehaviorBringCubeToBeacon",
    "BehaviorVisitInterestingEdge",
    "BehaviorExploreVisitPossibleMarker",
]


class BeaconMap:
    """ The beacon areas a robot has been, per robot, forgotten with it. """

    def __init__(self) -> None:
        self.areas: List[Tuple[float, float, float]] = []
        self.last_failure: float = 0.0


_beacon_maps: "weakref.WeakKeyDictionary[Any, BeaconMap]" = weakref.WeakKeyDictionary()


def beacons_for(cli: Any) -> BeaconMap:
    """ The beacon map for a robot, starting empty. """
    try:
        return _beacon_maps[cli]
    except (KeyError, TypeError):
        fresh = BeaconMap()
        try:
            _beacon_maps[cli] = fresh
        except TypeError:
            pass
        return fresh


def _has_gone_somewhere(cli: Any, radius_mm: float) -> bool:
    """
    Whether the robot has driven further than the radius from where its odometry starts.

    The pose starts at the origin and only driving moves it, so a robot that has gone nowhere
    sits within any radius of it. Thinking about places, and looking over ones already passed,
    are for after going somewhere; a robot that has not is fresh, and holds back.
    """
    try:
        position = cli.pose.position
        return math.hypot(position.x, position.y) > radius_mm
    except Exception:
        return False


class BehaviorLookAround(BehaviorScript):
    """
    The staged look-round the hiking scans share: the body turns through
    the ranges the configuration lays out while the head sweeps its own,
    pausing on the wait trigger between times. The camera stays on so the
    looking feeds the trackers: cubes, faces and markers seen on the way
    are what the scan is for. Recent-location bookkeeping belongs to the
    memory map and is not here.
    """

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        params = conf.get("params", {})
        self.scans = int(params.get("behavior_NumberOfScansBeforeStop", 1))
        self.clockwise_chance = float(params.get("s0_MainTurnCWChance", 0.5))
        self.body_speed = math.radians(float(params.get("sx_BodyTurnSpeed_degPerSec", 120.0)))
        self.head_speed = math.radians(float(params.get("sxh_HeadTurnSpeed_degPerSec", 60.0)))
        self.body_range = (float(params.get("s1_BodyAngleRangeMin_deg", 10.0)),
                           float(params.get("s1_BodyAngleRangeMax_deg", 30.0)))
        self.head_range = (float(params.get("s1_HeadAngleRangeMin_deg", -15.0)),
                           float(params.get("s1_HeadAngleRangeMax_deg", -5.0)))
        self.wait_range = (float(params.get("s2_WaitMin_sec", 0.5)),
                           float(params.get("s2_WaitMax_sec", 1.25)))
        self.wait_trigger = str(params.get("s2_WaitAnimTrigger", ""))
        self.head_moves = (int(params.get("s4_HeadAngleChangesMin", 2)),
                           int(params.get("s4_HeadAngleChangesMax", 4)))
        self.head_pause = (float(params.get("s4_WaitBetweenChangesMin_sec", 0.1)),
                           float(params.get("s4_WaitBetweenChangesMax_sec", 0.5)))
        self.last_clockwise: Optional[bool] = None

    def wants_to_run(self) -> bool:
        if not self.conf.get("params", {}).get("behavior_CanCarryCube", True):
            cubes = getattr(self.cli, "cubes", None)
            if cubes is not None and getattr(cubes, "carried", None) is not None:
                return False
        return True

    def _turn(self, degrees: float, cancel: threading.Event) -> None:
        _check(cancel)
        self.cli.turn_in_place(util.Angle(degrees=degrees), speed=self.body_speed, wait=True)
        _check(cancel)

    def _look(self, degrees: float, cancel: threading.Event) -> None:
        _check(cancel)
        self.cli.set_head_angle(math.radians(degrees))
        _check(cancel)

    def _pause(self, cancel: threading.Event) -> None:
        _check(cancel)
        if self.wait_trigger:
            self.play(self.wait_trigger, cancel)
        if cancel.wait(random.uniform(*self.wait_range)):
            raise Cancelled()

    def script(self, cancel: threading.Event) -> None:
        self.cli.enable_camera(True, color=False)
        if self.conf.get("params", {}).get("behavior_ShouldLowerLift", False):
            self.lower_lift(cancel)
        clockwise = random.random() < self.clockwise_chance
        if self.conf.get("params", {}).get("behavior_ShouldResetTurnDirection", False):
            if self.last_clockwise is not None:
                clockwise = not self.last_clockwise
            self.last_clockwise = clockwise
        direction = -1.0 if clockwise else 1.0
        for _ in range(max(1, self.scans)):
            turned = 0.0
            while turned < 360.0:
                _check(cancel)
                step = min(random.uniform(*self.body_range), 360.0 - turned)
                self._turn(direction * step, cancel)
                turned += step
                self._look(random.uniform(*self.head_range), cancel)
                self._pause(cancel)
            for _ in range(random.randint(*self.head_moves)):
                self._look(random.uniform(*self.head_range), cancel)
                if cancel.wait(random.uniform(*self.head_pause)):
                    raise Cancelled()


class BehaviorExploreLookAroundInPlace(BehaviorLookAround):
    """
    ExploreLookAroundInPlace - scan the world for cubes and edges.

    Runs in the general case of having nothing interesting at the moment:
    a full look round so that what is seen can trigger other behaviors.
    """


class BehaviorLookInPlaceMemoryMap(BehaviorLookAround):
    """
    LookInPlaceMemoryMap - look round to clear up unknown areas.

    Same scan as exploring, aimed by its own configuration at the unknown
    near the robot. Without a memory map the looking is the whole of it.
    It holds back until the robot has gone somewhere to look over: past
    the configuration's location distance from where it started.
    """

    def wants_to_run(self) -> bool:
        if not super().wants_to_run():
            return False
        threshold = float(self.conf.get("params", {}).get(
            "distanceThresholdForLocations_mm", 30.0))
        return _has_gone_somewhere(self.cli, threshold)


class BehaviorThinkAboutBeacons(BehaviorScript):
    """
    ThinkAboutBeacons - decide about beacons, showing intent meanwhile.

    Remembers the ground the robot stands on as a beacon area, and plays
    the new-area animation when it is somewhere it has not been: that is
    the thinking the configuration asks for, made visible.
    """

    def wants_to_run(self) -> bool:
        beacons = beacons_for(self.cli)
        if beacons.areas:
            return True
        radius = float(self.conf.get("params", {}).get("beaconRadius_mm", 175.0))
        return _has_gone_somewhere(self.cli, radius)

    def script(self, cancel: threading.Event) -> None:
        params = self.conf.get("params", {})
        radius = float(params.get("beaconRadius_mm", 175.0))
        trigger = str(params.get("newAreaAnimTrigger", ""))
        beacons = beacons_for(self.cli)
        x, y = self.cli.pose.position.x, self.cli.pose.position.y
        for ax, ay, _ in beacons.areas:
            if math.hypot(x - ax, y - ay) <= radius:
                return
        beacons.areas.append((x, y, radius))
        if trigger:
            self.play(trigger, cancel)


class BehaviorBringCubeToBeacon(BehaviorScript):
    """
    BringCubeToBeacon - fetch a cube into a beacon area.

    Picks up the nearest usable cube unless one is carried already, drives
    it to the nearest beacon the robot has thought about, and puts it down
    there. A failed run holds later ones off for the cooldown the
    configuration asks for.
    """

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.cooldown = float(conf.get("params", {}).get("recentFailureCooldown_sec", 45.0))
        self.last_failure: float = 0.0

    def wants_to_run(self) -> bool:
        if time.perf_counter() - self.last_failure < self.cooldown:
            return False
        cubes = getattr(self.cli, "cubes", None)
        if cubes is not None and getattr(cubes, "carried", None) is not None:
            return True
        try:
            return bool(usable_cubes(self.cli))
        except Exception:
            return False

    def _fail(self) -> None:
        self.last_failure = time.perf_counter()
        self.give_up("could not bring a cube to a beacon")

    def script(self, cancel: threading.Event) -> None:
        beacons = beacons_for(self.cli)
        x, y = self.cli.pose.position.x, self.cli.pose.position.y
        if not beacons.areas:
            return self._fail()
        target = min(beacons.areas, key=lambda a: math.hypot(x - a[0], y - a[1]))
        cubes = getattr(self.cli, "cubes", None)
        if cubes is None or getattr(cubes, "carried", None) is None:
            try:
                choices = usable_cubes(self.cli)
            except Exception:
                choices = []
            if not choices:
                return self._fail()
            if not self.pick_up(choices[0], cancel):
                return self._fail()
        bearing = math.atan2(target[1] - y, target[0] - x)
        heading = self.cli.pose.rotation.angle_z.radians
        self.cli.turn_in_place(util.Angle(radians=_wrap(bearing - heading)), wait=True)
        _check(cancel)
        if not self.cli.go_to_pose(util.Pose(target[0], target[1], 0.0,
                                             angle_z=util.Angle(radians=bearing)), wait=True):
            return self._fail()
        if not cube_handling.put_down_cube(self.cli, cancel=cancel):
            return self._fail()
        self.need_action()


class BehaviorVisitInterestingEdge(BehaviorScript):
    """
    VisitInterestingEdge - go and have a proper look at an edge.

    Squints, edges a little closer, and has a proper observe: the vantage
    geometry belongs to the memory map and is not here, so it approaches
    straight and lets the cliff stop keep it honest. A drive that gives up
    halfway means no edge was there to visit after all. Cubes are what
    counts as interesting here; with none known there is nothing to visit.
    """

    def wants_to_run(self) -> bool:
        try:
            return bool(usable_cubes(self.cli))
        except Exception:
            return False

    def script(self, cancel: threading.Event) -> None:
        params = self.conf.get("params", {})
        self.cli.enable_camera(True, color=False)
        self.cli.set_head_angle(math.radians(-5.0))
        self.play(str(params.get("squintStartAnimTrigger", "HikingSquintStart")), cancel)
        _check(cancel)
        speed = float(params.get("borderApproachSpeed_mmps", 50.0))
        if not self.cli.drive_straight(util.Distance(mm=60.0), speed=speed, wait=True):
            self.play(str(params.get("edgesNotFoundAnimTrigger", "")), cancel)
            return
        self.play(str(params.get("observeEdgeAnimTrigger", "HikingObserve")), cancel)
        self.play(str(params.get("squintEndAnimTrigger", "HikingSquintEnd")), cancel)


class BehaviorExploreVisitPossibleMarker(BehaviorScript):
    """
    ExploreVisitPossibleMarker - go and confirm something half-seen.

    Drives to within looking distance of the nearest cube or remembered
    charger, which is what visiting a possible marker amounts to without
    the marker tracker. Nothing known means nothing to visit.
    """

    #: How near to get, in mm.
    VISIT_DISTANCE = 200.0

    def wants_to_run(self) -> bool:
        return bool(self._candidates())

    def _candidates(self) -> List[Tuple[float, float]]:
        found: List[Tuple[float, float]] = []
        try:
            for cube in usable_cubes(self.cli):
                if cube.pose is not None:
                    found.append((cube.pose.x, cube.pose.y))
        except Exception:
            pass
        charger = getattr(self.cli, "charger", None)
        pose = getattr(charger, "pose", None) if charger is not None else None
        if pose is not None:
            found.append((pose.position.x, pose.position.y))
        return found

    def script(self, cancel: threading.Event) -> None:
        x, y = self.cli.pose.position.x, self.cli.pose.position.y
        found = self._candidates()
        if not found:
            return self.give_up("nothing to visit")
        tx, ty = min(found, key=lambda p: math.hypot(x - p[0], y - p[1]))
        dist = math.hypot(x - tx, y - ty)
        if dist <= self.VISIT_DISTANCE:
            return
        bearing = math.atan2(ty - y, tx - x)
        heading = self.cli.pose.rotation.angle_z.radians
        self.cli.turn_in_place(util.Angle(radians=_wrap(bearing - heading)), wait=True)
        _check(cancel)
        stand = max(0.0, dist - self.VISIT_DISTANCE)
        px = x + stand * math.cos(bearing)
        py = y + stand * math.sin(bearing)
        if not self.cli.go_to_pose(util.Pose(px, py, 0.0, angle_z=util.Angle(radians=bearing)),
                                   wait=True):
            return self.give_up("could not get there")


def find_laser_spot(image: Any) -> Optional[Tuple[float, float, int]]:
    """
    Where the brightest blob in a frame is, if it looks like a laser dot.

    A laser pointer saturates a small patch however the exposure is set, so
    saturated pixels are the whole test: a few to a few thousand of them,
    and not half the frame, which would be a lamp or a window. Returns the
    middle of the patch in pixels and how many of them there are.
    """
    import numpy as np
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    rows, cols = np.nonzero(gray >= 250)
    if len(rows) < 3 or len(rows) > gray.size // 4:
        return None
    return float(cols.mean()), float(rows.mean()), len(rows)


class BehaviorTrackLaser(BehaviorScript):
    """
    TrackLaser - find a laser dot, follow it, and pounce on it.

    Scans round for something saturated enough to be a laser, turns to
    keep it centered, and lunges when it sits still ahead. Scans that find
    nothing count against the configured failures, and three in a row end
    it. The darkened exposure the configuration asks for needs an exposure
    control PyCozmo does not have yet; a laser saturates at any exposure,
    so the hunt works regardless.
    """

    #: How far the lunge goes, in mm, and how fast, in mm/s. A spot counts as
    #: centered when it bears less than five degrees off ahead - about 25 px.
    CENTER_TOLERANCE = math.radians(5.0)
    LUNGE_DISTANCE = 150.0
    LUNGE_SPEED = 150.0
    #: Looks per scan round, and scans that may find nothing in a row.
    SCAN_STEPS = 12
    SCAN_TURN = math.radians(30.0)

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        params = conf.get("params", {})
        self.max_failures = int(params.get("maxFailures", 3))
        self.look_around_trigger = str(params.get("lookAroundAnimTrigger", ""))
        self.look_down_trigger = str(params.get("lookInPlaceAnimTrigger", ""))

    def wants_to_run(self) -> bool:
        return True

    def _bearing(self, x: float) -> float:
        calib = getattr(self.cli, "camera_calibration", None) or camera.DEFAULT_CALIBRATION
        return math.atan((x - calib.cx) / calib.fx)

    def _snap(self) -> Optional[Tuple[float, float, int]]:
        images: List[Any] = []
        got = threading.Event()

        def on_image(_cli: Any, image: Any) -> None:
            images.append(image)
            got.set()

        handler = self.cli.add_handler(event.EvtNewRawCameraImage, on_image, one_shot=True)
        try:
            if not got.wait(2.0):
                return None
        finally:
            self.cli.del_handler(event.EvtNewRawCameraImage, handler)
        return find_laser_spot(images[0])

    def script(self, cancel: threading.Event) -> None:
        self.cli.enable_camera(True, color=False)
        self.cli.set_head_angle(math.radians(-5.0))
        failures = 0
        lunges = 0
        while failures < self.max_failures and lunges < 3:
            _check(cancel)
            if self.look_around_trigger:
                self.play(self.look_around_trigger, cancel)
            spot = None
            for _ in range(self.SCAN_STEPS):
                _check(cancel)
                spot = self._snap()
                if spot is not None:
                    break
                self.cli.turn_in_place(util.Angle(radians=self.SCAN_TURN), wait=True)
            if spot is None:
                failures += 1
                if self.look_down_trigger:
                    self.play(self.look_down_trigger, cancel)
                continue
            failures = 0
            self.play("LaserAcknowledge", cancel)
            x, _, _ = spot
            bearing = self._bearing(x)
            if abs(bearing) > self.CENTER_TOLERANCE:
                self.cli.turn_in_place(util.Angle(radians=bearing), wait=True)
                _check(cancel)
                continue
            self.play("LaserPounce", cancel)
            self.play("LaserDriveStart", cancel)
            self.cli.drive_straight(util.Distance(mm=self.LUNGE_DISTANCE),
                                    speed=self.LUNGE_SPEED, wait=True)
            self.play("LaserDriveEnd", cancel)
            lunges += 1
            _check(cancel)
        self.play("LaserGetOut", cancel)
