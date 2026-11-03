"""

Feeding behaviors: search for a cube to eat from, then eat from it.

Eating here means what the triggers say: drive the fork to the cube, lay
it on top, and nibble while watching for the cube to be shaken - a shake
counts as a feed. The severe variants of each step play when the energy
need is critical, the normal ones otherwise.

"""

import threading
from typing import Any

from . import cube_handling, util
from .cube_behaviors import BehaviorScript, usable_cubes
from .cube_handling import Cancelled, _check
from .needs import BRACKETS


__all__ = [
    "BehaviorFeedingSearchForCube",
    "BehaviorFeedingEat",
]


def _is_severe(needs: Any) -> bool:
    """ Whether the energy need is critical, asking for the severe variant of each step. """
    if needs is not None:
        try:
            return bool(needs.bracket("Energy") == BRACKETS[-1])
        except Exception:
            pass
    return False


def _paired(base: str, needs: Any) -> str:
    """ The trigger with its _Severe or _Normal suffix, as AnimationTriggerMap.json names them. """
    return base + ("_Severe" if _is_severe(needs) else "_Normal")


def _requested(base: str, needs: Any) -> str:
    """ The ask-for trigger, bare normally and _Severe when energy is critical. """
    return base + ("_Severe" if _is_severe(needs) else "")


class BehaviorFeedingSearchForCube(BehaviorScript):
    """
    FeedingSearchForCube - turn round looking for a cube to eat from.

    Asks out loud first, stops the moment a usable cube is in the tracker,
    and admits failure when a full turn finds nothing.
    """

    #: How far the robot turns between looks, in radians.
    TURN_STEP = 0.5235987755982988  # 30 degrees

    def wants_to_run(self) -> bool:
        return True

    def script(self, cancel: threading.Event) -> None:
        self.cli.enable_camera(True, color=False)
        self.play(_requested("FeedingSearchRequest", self.needs), cancel)
        if usable_cubes(self.cli):
            self.play(_paired("FeedingReactToSeeCube", self.needs), cancel)
            return
        steps = int(round(2.0 * 3.141592653589793 / self.TURN_STEP))
        for _ in range(steps):
            _check(cancel)
            self.cli.turn_in_place(util.Angle(radians=self.TURN_STEP), wait=True)
            if cancel.wait(0.3):
                raise Cancelled()
            if usable_cubes(self.cli):
                self.play(_paired("FeedingReactToSeeCube", self.needs), cancel)
                return
        self.play(_requested("FeedingSearchFailure", self.needs), cancel)


class BehaviorFeedingEat(BehaviorScript):
    """
    FeedingEat - drive the fork to a cube, lay it on, and nibble.

    The cube getting shaken while the fork is on it counts as a feed;
    sitting still until the watch runs out counts as not fed.
    """

    #: How long a shake is waited for, in seconds, and what counts as one, in mm.
    WATCH_TIMEOUT = 8.0
    SHAKE_THRESHOLD = 15.0
    #: Nibbles before and after the watch.
    NIBBLE_DISTANCE = 20.0
    NIBBLE_SPEED = 40.0

    def __init__(self, cli: Any, conf: Any, robot_needs: Any = None):
        super().__init__(cli, conf, robot_needs)
        self.watch_timeout = float(conf.get("watchForShake_sec", self.WATCH_TIMEOUT))

    def wants_to_run(self) -> bool:
        return bool(usable_cubes(self.cli))

    def _pose_of(self, cube: Any) -> Any:
        return cube.pose

    def script(self, cancel: threading.Event) -> None:
        cubes = usable_cubes(self.cli)
        if not cubes:
            return
        cube = cubes[0]
        self.cli.enable_camera(True, color=False)
        if not cube_handling.go_to_cube(self.cli, cube, cancel=cancel):
            return self.give_up("could not get to the cube")
        self.cli.drive_straight(util.Distance(mm=self.NIBBLE_DISTANCE),
                                speed=self.NIBBLE_SPEED, wait=True)
        _check(cancel)
        self.play(_paired("FeedingPlaceLiftOnCube", self.needs), cancel)
        before = self._pose_of(cube)
        shaken = False
        waited = 0.0
        while waited < self.watch_timeout:
            _check(cancel)
            if cancel.wait(0.2):
                raise Cancelled()
            waited += 0.2
            now = self._pose_of(cube)
            if before is not None and now is not None and \
                    abs(now.x - before.x) + abs(now.y - before.y) + abs(now.z - before.z) \
                    > self.SHAKE_THRESHOLD:
                shaken = True
                break
        if shaken:
            self.play(_paired("FeedingReactToShake", self.needs), cancel)
            self.cli.drive_straight(util.Distance(mm=self.NIBBLE_DISTANCE),
                                    speed=self.NIBBLE_SPEED, wait=True)
            self.play(_paired("FeedingAteFullEnough", self.needs), cancel)
        else:
            self.play(_paired("FeedingAteNotFullEnough", self.needs), cancel)
