"""

Player-facing freeplay: the bouncer's tap game, the guard-dog cube watch, the fire-truck alarm spark, and the
onboarding cube showing.

Bouncer asks to play, with the idea and the request it names, and waits for a cube tap to take it up; silence
times it out. Each tap after that scores, celebrated up to three, and then it is done. The scoring rounds are not
spelled out in the configuration - IdeaToPlay, RequestToPlay, Wait, Timeout, IntoScore1-3, GetOut are all it names -
so taps-to-score is the reading that uses every trigger.

GuardDog guards the cubes it knows, asleep: settling down, then sleeping in rounds with a pulse every few. A cube
that moves wakes it; a face in view means the player is busted, a cube turned over unseen means the player scored,
and anything else was a false alarm and it settles back down. A cube that disconnects, or the watch running out,
ends it, naming whether any cube was touched. The rules are the minigame's own: turn all the cubes over without
waking him - he senses them moving.

FireTruckAlarm has no animation of its own in the resources - the trigger map names none, and no siren clip either -
so it is built from what the robot can do: a two-tone siren it synthesizes, flashing red backpack lights, and an
alarm circuit driven round. OnboardingShowCube finds a cube, picks it up to show it, and waits for the player to take
it, counting failed finds and pickups against the configuration's budgets.

"""

import math
import struct
import tempfile
import threading
import time
import wave
from typing import Any, List, Optional, Tuple

from . import cube_handling
from . import event
from . import lights
from . import robot
from . import util
from .cube_behaviors import BehaviorScript, usable_cubes
from .cube_handling import Cancelled, _check
from .cubes import LightCube


__all__ = [
    "BehaviorBouncer",
    "BehaviorGuardDog",
    "BehaviorFireTruckAlarm",
    "BehaviorOnboardingShowCube",
]


def _wait_for_tap(cli: Any, cubes: List[LightCube], timeout: float,
                  cancel: threading.Event, play: Any = None, idle: Optional[str] = None
                  ) -> Optional[LightCube]:
    """
    The first of the cubes tapped before the timeout, or None. While waiting, the idle animation plays through the
    play function given, the way the game requests idle while the player answers.
    """
    tapped: List[LightCube] = []
    answered = threading.Event()

    def on_tapped(_: Any, cube: LightCube, taps: int) -> None:
        if cube in cubes and not answered.is_set():
            tapped.append(cube)
            answered.set()

    handler = cli.add_handler(event.EvtCubeTapped, on_tapped)
    try:
        deadline = time.perf_counter() + timeout
        while not answered.is_set() and time.perf_counter() < deadline:
            _check(cancel)
            if play is not None and idle is not None:
                play(idle, cancel)
            if answered.wait(0.2):
                break
    finally:
        cli.del_handler(event.EvtCubeTapped, handler)
    return tapped[0] if tapped else None


class BehaviorBouncer(BehaviorScript):
    """
    Bouncer - ask for a tap game, and count the taps that follow as scores.

    The idea, the request, and the wait for an answer are the configuration's; a tap takes the game up and each tap
    after scores, celebrated to three, and silence at any wait ends it.
    """

    #: How long the answer and each score are waited for, in seconds, and how many scores end the game.
    REQUEST_TIMEOUT = 15.0
    SCORE_TIMEOUT = 10.0
    MAX_SCORES = 3

    def wants_to_run(self) -> bool:
        return any(cube.connected for cube in self.cli.cubes)

    def script(self, cancel: threading.Event) -> None:
        request_timeout = float(self.conf.get("requestTimeout_sec", self.REQUEST_TIMEOUT))
        score_timeout = float(self.conf.get("scoreTimeout_sec", self.SCORE_TIMEOUT))
        cubes = [cube for cube in self.cli.cubes if cube.connected]
        self.play("BouncerIdeaToPlay", cancel)
        self.play("BouncerRequestToPlay", cancel)
        if _wait_for_tap(self.cli, cubes, request_timeout, cancel,
                         play=self.play, idle="BouncerWait") is None:
            self.play("BouncerTimeout", cancel)
            self.play("BouncerGetOut", cancel)
            return
        for score in range(1, self.MAX_SCORES + 1):
            _check(cancel)
            if _wait_for_tap(self.cli, cubes, score_timeout, cancel,
                             play=self.play, idle="BouncerWait") is None:
                break
            self.play("BouncerIntoScore{}".format(score), cancel)
        self.play("BouncerGetOut", cancel)


def _pose_of(cube: Any) -> Optional[Tuple[float, float, float, Any]]:
    """ Where a cube is and which way is up, or None when it is not known. """
    pose = cube.pose
    if pose is None:
        return None
    return (pose.x, pose.y, pose.z, cube.up_axis)


class BehaviorGuardDog(BehaviorScript):
    """
    GuardDog - sleep by the cubes, and wake when one moves.

    Settling down, sleeping in rounds with a pulse every few, watching the cubes it knew when it settled. A cube that
    disconnects ends it at once; one that moves wakes it. A face in view then means the player is busted; a cube
    turned over with nobody seen means the player scored; anything else was a false alarm and it sleeps on. The watch
    running out names whether any cube was touched.
    """

    #: How far a cube may move unnoticed, in mm; a turned-over cube always counts, however far.
    MOVE_THRESHOLD = 10.0
    #: How long each sleeping round watches, in seconds, checking that often, and every how many rounds the pulse plays.
    ROUND_SECONDS = 2.0
    SAMPLE_INTERVAL = 0.2
    PULSE_EVERY = 3
    #: How long the whole watch lasts, in seconds.
    WATCH_TIMEOUT = 90.0

    def wants_to_run(self) -> bool:
        return bool(usable_cubes(self.cli))

    def _snapshot(self) -> List[Tuple[Any, Tuple[float, float, float, Any]]]:
        """ The cubes known now, each with where it is. """
        seen = []
        for cube in self.cli.cubes:
            if cube.connected:
                pose = _pose_of(cube)
                if pose is not None:
                    seen.append((cube, pose))
        return seen

    def _moved(self, before: Tuple[float, float, float, Any], cube: Any) -> bool:
        """ Whether the cube has moved since the snapshot: gone far, or turned. """
        now = _pose_of(cube)
        if now is None:
            return True
        if now[3] != before[3]:
            return True
        return math.hypot(now[0] - before[0], now[1] - before[1], now[2] - before[2]) > self.MOVE_THRESHOLD

    def _flipped(self, before: Tuple[float, float, float, Any], cube: Any) -> bool:
        """ Whether the cube was turned over since the snapshot: up is no longer up. """
        now = _pose_of(cube)
        if now is None:
            return False
        return now[3] != before[3] and now[3] is not None

    def script(self, cancel: threading.Event) -> None:
        watch_timeout = float(self.conf.get("timeout_sec", self.WATCH_TIMEOUT))
        round_seconds = float(self.conf.get("round_sec", self.ROUND_SECONDS))
        self.play("GuardDogSettle", cancel)
        baseline = self._snapshot()
        touched = False
        rounds = 0
        deadline = time.perf_counter() + watch_timeout
        while time.perf_counter() < deadline:
            _check(cancel)
            if rounds % self.PULSE_EVERY == self.PULSE_EVERY - 1:
                self.play("GuardDogPulse", cancel)
            else:
                self.play("GuardDogSleepLoop", cancel)
            rounds += 1
            woke = self._watch(round_seconds, baseline, cancel)
            if woke is None:
                continue
            cube, _ = woke
            touched = True
            if not cube.connected:
                self.play("GuardDogCubeDisconnect", cancel)
                return
            if self.cli.faces.visible():
                self.play("GuardDogBusted", cancel)
                return
            if self._flipped(dict(baseline).get(cube, (0.0, 0.0, 0.0, None)), cube):
                self.play("GuardDogPlayerSuccess", cancel)
                return
            self.play("GuardDogFakeout", cancel)
            baseline = self._snapshot()
        self.play("GuardDogTimeoutCubesTouched" if touched else "GuardDogTimeoutCubesUntouched", cancel)

    def _watch(self, seconds: float, baseline: List[Tuple[Any, Tuple[float, float, float, Any]]],
               cancel: threading.Event) -> Optional[Tuple[Any, Tuple[float, float, float, Any]]]:
        """ The first baseline cube that moves within the seconds, or None. """
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            _check(cancel)
            for cube, before in baseline:
                if not cube.connected or self._moved(before, cube):
                    return (cube, before)
            if cancel.wait(self.SAMPLE_INTERVAL):
                raise Cancelled()
        return None


def siren_wav(path: str, cycles: int = 3) -> str:
    """
    Write a two-tone wee-woo siren to the path, and return it: 16-bit mono at 22050 Hz, which is what the robot
    plays. Each cycle wails low-high-low; there is no siren clip in the resources to play instead.
    """
    rate = 22050
    low, high = 700.0, 950.0
    tone = 0.35
    frames = bytearray()
    for _ in range(cycles):
        for freq in (low, high, low):
            count = int(rate * tone)
            for i in range(count):
                sample = int(16000.0 * math.sin(2.0 * math.pi * freq * i / rate))
                frames += struct.pack("<h", sample)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))
    return path


class BehaviorFireTruckAlarm(BehaviorScript):
    """
    FireTruckAlarm - sound the alarm: siren, flashing lights, and an alarm circuit.

    The user's spark, so it always wants to run. Each round wails the siren, flashes the backpack red, and drives a
    small square; the activity's thirty seconds are a few such rounds.
    """

    #: How many alarm rounds make the performance, and how far each leg of the square is, in mm.
    ROUNDS = 3
    LEG_DISTANCE = 100.0
    LEG_SPEED = 80.0

    def wants_to_run(self) -> bool:
        return True

    def script(self, cancel: threading.Event) -> None:
        rounds = int(self.conf.get("rounds", self.ROUNDS))
        with tempfile.TemporaryDirectory(prefix="firetruck") as directory:
            siren = siren_wav(directory + "/siren.wav")
            for _ in range(rounds):
                _check(cancel)
                self.cli.play_audio(siren)
                for _ in range(4):
                    _check(cancel)
                    self.cli.set_all_backpack_lights(lights.red_light)
                    self.cli.drive_straight(util.Distance(mm=self.LEG_DISTANCE),
                                            speed=self.LEG_SPEED, wait=True)
                    _check(cancel)
                    self.cli.set_backpack_lights_off()
                    self.cli.turn_in_place(util.Angle(degrees=90.0), wait=True)
        self.cli.set_backpack_lights_off()


class BehaviorOnboardingShowCube(BehaviorScript):
    """
    OnboardingShowCube - find a cube, pick it up to show it, and hand it over.

    Looks for a cube it knows until the find budget runs out, drives to it, and picks it up until the pickup budget
    does; lifts it high to show it, and waits for the player's tap before putting it back down. The whole showing
    ends with the configuration's timeout.
    """

    #: How many failed finds and pickups are borne, and how long the whole showing lasts, in seconds.
    MAX_ERRORS_TOTAL = 3
    MAX_ERRORS_PICKUP = 3
    TIMEOUT = 300.0

    def wants_to_run(self) -> bool:
        return True

    def script(self, cancel: threading.Event) -> None:
        max_errors = int(self.conf.get("MaxErrorsTotal", self.MAX_ERRORS_TOTAL))
        max_pickup_errors = int(self.conf.get("MaxErrorsPickup", self.MAX_ERRORS_PICKUP))
        timeout = float(self.conf.get("Timeout_Sec", self.TIMEOUT))
        deadline = time.perf_counter() + timeout
        errors = 0
        pickup_errors = 0
        while time.perf_counter() < deadline:
            _check(cancel)
            cubes = [cube for cube in self.cli.cubes if cube.connected]
            if not cubes or not cube_handling.find_cube(self.cli, cubes[0], cancel=cancel):
                errors += 1
                if errors > max_errors:
                    return self.give_up("found no cube")
                continue
            self.play("OnboardingDiscoverCube", cancel)
            cube = cubes[0]
            if not cube_handling.go_to_cube(self.cli, cube, cancel=cancel):
                self.play("OnboardingCubeDockFail", cancel)
                errors += 1
                if errors > max_errors:
                    return self.give_up("got to no cube")
                continue
            self.play("OnboardingReactToCube", cancel)
            if not cube_handling.pick_up_cube(self.cli, cube, cancel=cancel):
                pickup_errors += 1
                if pickup_errors > max_pickup_errors:
                    return self.give_up("picked up no cube")
                continue
            self.play("OnboardingSoundOnlyLiftEffortPickup", cancel)
            self.cli.set_lift_height(robot.MAX_LIFT_HEIGHT.mm)
            _check(cancel)
            self.play("OnboardingInteractWithCube", cancel)
            remaining = deadline - time.perf_counter()
            _wait_for_tap(self.cli, [cube], max(0.0, remaining), cancel)
            self.play("OnboardingSoundOnlyLiftEffortPlaceLow", cancel)
            cube_handling.put_down_cube(self.cli, cube, cancel=cancel)
            self.play("OnboardingReactToCubePutDown", cancel)
            return
        return self.give_up("ran out of time")
