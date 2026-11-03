"""

Face behaviors: finding faces, driving to them, enrolling them, interacting with them, peek-a-boo,
and the put-down dispatch that looks for a face and a cube.

Faces come from the client's face tracker, which the brain feeds with camera images: see
pycozmo.faces and pycozmo.face_detection. Finding them needs OpenCV and the face models -
tools/pycozmo_faces.py download - without which the behaviors hold back.

"""

import math
import threading
import time
from typing import Any, List, Optional

from . import util
from .cube_behaviors import BehaviorScript
from .cube_handling import _check, _wrap
from .logger import logger


__all__ = [
    "BehaviorFindFaces",
    "BehaviorSearchForFace",
    "BehaviorDriveToFace",
    "BehaviorEnrollFace",
    "BehaviorInteractWithFaces",
    "BehaviorPeekABoo",
    "BehaviorLookForFaceAndCube",
]


def _fresh_faces(cli: Any, max_age_s: float) -> List[Any]:
    """ The faces in sight, seen within the last max_age_s seconds, nearest first. """
    now = time.perf_counter()
    return [face for face in cli.faces.visible() if face.seen_within(max_age_s, now)]


def _face_bearing(cli: Any, face: Any) -> Optional[float]:
    """ Which way a face is from where the robot looks, left for positive, in radians. """
    if face.pose is None:
        return None
    dx = face.pose.x - cli.pose.position.x
    dy = face.pose.y - cli.pose.position.y
    return _wrap(math.atan2(dy, dx) - cli.pose.rotation.angle_z.radians)


class BehaviorFindFaces(BehaviorScript):
    """
    FindFaces - turn round looking for a face.

    Looks in the direction the robot faces first, so a face right ahead is
    found without turning, then round in steps until one is seen. Finding
    one ends the behavior at once: what happens with it is the next
    behavior's business.
    """

    #: How far the robot turns between looks, in radians, and how long it settles first.
    TURN_STEP = math.radians(30.0)
    SETTLE_TIME = 0.3

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.max_face_age = float(conf.get("maxFaceAgeToLook_ms", 10000.0)) / 1000.0

    def wants_to_run(self) -> bool:
        return self.cli.faces.available()

    def script(self, cancel: threading.Event) -> None:
        if _fresh_faces(self.cli, self.max_face_age):
            return
        self.cli.enable_camera(True, color=False)
        self.cli.set_head_angle(0.0)
        if cancel.wait(self.SETTLE_TIME):
            return
        steps = int(round(2.0 * math.pi / self.TURN_STEP))
        for _ in range(steps):
            _check(cancel)
            self.cli.turn_in_place(util.Angle(radians=self.TURN_STEP), wait=True)
            if cancel.wait(self.SETTLE_TIME):
                return
            if _fresh_faces(self.cli, self.max_face_age):
                return


class BehaviorSearchForFace(BehaviorFindFaces):
    """
    SearchForFace - the voice-command variant of finding faces.

    A spoken "come here" starts with a look round, which is what finding
    faces does; driving to the face found is DriveToFace's business.
    """


class BehaviorDriveToFace(BehaviorScript):
    """
    DriveToFace - go to the face that was asked to come here.

    Turns to the nearest face in sight and drives over in short legs,
    stopping at a social distance. Already there means saying so instead
    of driving at all.
    """

    #: How near is near enough, in mm, and how far one leg drives at most.
    STOP_DISTANCE = 250.0
    LEG_DISTANCE = 100.0
    #: How far off the face the robot may be and still drive straight at it, in radians.
    AIM_TOLERANCE = math.radians(5.0)
    #: How many legs it drives at most, and how long a leg may take.
    MAX_LEGS = 20

    def wants_to_run(self) -> bool:
        return self.cli.faces.available() and bool(self.cli.faces.visible())

    def script(self, cancel: threading.Event) -> None:
        faces = _fresh_faces(self.cli, self.max_face_age)
        if not faces:
            return
        face = faces[0]
        if face.distance <= self.STOP_DISTANCE:
            self.play("ComeHere_AlreadyHere", cancel)
            return
        self.cli.enable_camera(True, color=False)
        for _ in range(self.MAX_LEGS):
            _check(cancel)
            faces = _fresh_faces(self.cli, self.max_face_age)
            if not faces:
                return
            face = faces[0]
            if face.distance <= self.STOP_DISTANCE:
                break
            bearing = _face_bearing(self.cli, face)
            if bearing is None:
                return
            if abs(bearing) > self.AIM_TOLERANCE:
                self.cli.turn_in_place(util.Angle(radians=bearing), wait=True)
                _check(cancel)
            leg = min(self.LEG_DISTANCE, face.distance - self.STOP_DISTANCE)
            if leg <= 0.0:
                break
            self.cli.drive_straight(util.Distance(mm=leg), wait=True)
        self.play("ComeHere_SearchForFace_FoundFace", cancel)

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.max_face_age = float(conf.get("maxFaceAgeToLook_ms", 10000.0)) / 1000.0


class BehaviorEnrollFace(BehaviorScript):
    """
    EnrollFace - learn the face in front of the camera.

    Waits for exactly one face to hold still ahead - more than one is an
    error in the configuration's terms - and enrolls its views under a
    name, which the user can change afterwards: rename it with
    tools/pycozmo_faces.py, or have the robot ask in RespondToRenameFace.
    """

    #: The default name a face is enrolled under; the configuration may name one of its own.
    DEFAULT_NAME = "Friend"

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.max_faces = int(conf.get("maxFacesVisible", 1))
        self.too_many_timeout = float(conf.get("tooManyFacesTimeout_sec", 1.5))

    def wants_to_run(self) -> bool:
        return self.cli.faces.available()

    def script(self, cancel: threading.Event) -> None:
        self.cli.enable_camera(True, color=False)
        self.cli.set_head_angle(0.0)
        deadline = time.perf_counter() + self.too_many_timeout
        while True:
            _check(cancel)
            if len(self.cli.faces.visible()) <= self.max_faces:
                break
            if time.perf_counter() >= deadline:
                logger.warning("Behavior '{}' sees too many faces to enroll.".format(self.get_id()))
                return
            if cancel.wait(0.1):
                return
        name = str(self.conf.get("name", self.DEFAULT_NAME))
        if self.cli.faces.enroll(name, timeout=30.0, cancel=cancel):
            self.play("MeetCozmoFirstEnrollmentCelebration", cancel)


class BehaviorInteractWithFaces(BehaviorScript):
    """
    InteractWithFaces - give a face some attention.

    Follows the nearest face in sight with the body, up to a maximum, and
    no less than a minimum: a short hello is not worth starting. The turns
    are clamped small, the way the configuration asks.
    """

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        params = conf.get("params", {})
        self.min_time = float(params.get("minTimeToTrackFace_s", 8.0))
        self.max_time = float(params.get("maxTimeToTrackFace_s", 15.0))
        self.min_period = float(params.get("minClampPeriod_s", 0.2))
        self.max_period = float(params.get("maxClampPeriod_s", 0.7))

    def wants_to_run(self) -> bool:
        return self.cli.faces.available() and bool(self.cli.faces.visible())

    def script(self, cancel: threading.Event) -> None:
        self.cli.enable_camera(True, color=False)
        end = time.perf_counter() + self.max_time
        tracked_until = float("-inf")
        while time.perf_counter() < end:
            _check(cancel)
            faces = self.cli.faces.visible()
            if not faces:
                if time.perf_counter() - tracked_until > self.min_time or tracked_until < 0.0:
                    break
            else:
                tracked_until = time.perf_counter()
                bearing = _face_bearing(self.cli, faces[0])
                if bearing is not None and abs(bearing) > math.radians(5.0):
                    self.cli.turn_in_place(util.Angle(radians=bearing), wait=True)
            if cancel.wait(self.min_period):
                return
            _check(cancel)
            if cancel.wait(max(0.0, self.max_period - self.min_period)):
                return


class BehaviorPeekABoo(BehaviorScript):
    """
    PeekABoo - the peek-a-boo game.

    Gets in, then hides and surprises up to a few times while a face stays
    to play with, and gets out happy when one does and sad when nobody
    stayed. The need action the configuration names is credited for a game
    played through.
    """

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.min_times = int(conf.get("minTimesPeekBeforeQuit", 1))
        self.max_times = int(conf.get("maxTimesPeekBeforeQuit", 3))
        self.max_face_age = float(conf.get("maxTimeOldestFaceToConsider_Sec", 10.0))

    def wants_to_run(self) -> bool:
        return self.cli.faces.available() and bool(_fresh_faces(self.cli, self.max_face_age))

    def script(self, cancel: threading.Event) -> None:
        self.cli.enable_camera(True, color=False)
        self.play("PeekABooGetIn", cancel)
        rounds = 0
        while rounds < self.max_times:
            _check(cancel)
            if rounds >= self.min_times and not _fresh_faces(self.cli, self.max_face_age):
                break
            self.play("PeekABooShort", cancel)
            self.play("PeekABooSurprised", cancel)
            rounds += 1
        if _fresh_faces(self.cli, self.max_face_age):
            self.play("PeekABooGetOutHappy", cancel)
        else:
            self.play("PeekABooGetOutSad", cancel)
        self.need_action()


class BehaviorLookForFaceAndCube(BehaviorScript):
    """
    LookForFaceAndCube - the put-down dispatch: look round for a face and a cube.

    Sweeps the body with the head lowered, the way the configuration lays
    out, and stops at a face it knows by name, turning towards it: what
    runs next depends on who and what was found.
    """

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        params = conf.get("params", {})
        self.side_picks = int(params.get("face_sidePicks", 2))
        self.sweep_angle = math.radians(float(params.get("face_bodyAngleRelRangeMax_deg", 30.0)))

    def wants_to_run(self) -> bool:
        return True

    def _cubes_about(self) -> bool:
        if not hasattr(self.cli, "cubes"):
            return False
        try:
            from .cube_behaviors import usable_cubes
            return bool(usable_cubes(self.cli))
        except Exception:
            return False

    def script(self, cancel: threading.Event) -> None:
        self.cli.enable_camera(True, color=False)
        self.play(str(self.conf.get("lookInPlaceAnimTrigger", "HikingLookAround")), cancel)
        for _ in range(max(1, self.side_picks)):
            for direction in (1.0, -1.0):
                _check(cancel)
                self.cli.turn_in_place(util.Angle(radians=direction * self.sweep_angle), wait=True)
                if cancel.wait(0.3):
                    return
                for face in self.cli.faces.visible():
                    if face.name is not None:
                        bearing = _face_bearing(self.cli, face)
                        if bearing is not None:
                            self.cli.turn_in_place(util.Angle(radians=bearing), wait=True)
                        return
                if self._cubes_about():
                    return
