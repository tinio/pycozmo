import math
import os
import time
import unittest
from typing import Dict, List, Set, cast
from unittest import mock

from PIL import Image

import pycozmo
import pycozmo.explore_behaviors
import pycozmo.face_behaviors
import pycozmo.feeding_behaviors


class FakeConnection(pycozmo.event.Dispatcher):
    """ Stands in for Connection, which is what behaviors post EvtBehaviorDone to. """

    def __init__(self):
        super().__init__()
        self.events = []

    def post_event(self, evt, *args, **kwargs):
        self.events.append((evt, args))


class FakeClient(pycozmo.event.Dispatcher):
    """ Stands in for Client, recording the animations a behavior asks for. """

    #: Animation groups the fake knows about, which is every trigger the behaviors under test use.
    ANIMATION_GROUPS = (
        "AcknowledgeObject",
        "CubeMovedSense",
        "DizzyShakeStop",
        "FacePlantRoll",
        "FistBumpLeftHanging",
        "FistBumpRequestOnce",
        "FlipDownFromBack",
        "FrustratedByFailure",
        "GoToSleepGetIn",
        "GoToSleepSleeping",
        "PetDetectionShort",
        "PlacedOnCharger",
        "ReactToCliff",
        "ReactToImpact",
        "ReactToNewBlockSmall",
        "ReactToOnLeftSide",
        "ReactToOnRightSide",
        "ReactToPickup",
        "ReactToUnexpectedMovement",
        "SoundOnlyRamIntoBlock",
        "SparkGetIn",
        "BuildPyramidSuccess",
        "BuildPyramidReactToBase",
        "BuildPyramidThankUser",
        "EarnedSparks",
        "MeetCozmoRenameFaceSayName",
        "NeedsMildLowEnergyRequest",
        "NeedsSevereLowEnergyRequest",
        "NeedsMildLowPlayRequest",
        "NeedsSevereLowPlayRequest",
        "NeedsMildLowRepairRequest",
        "NeedsSevereLowRepairRequest",
        "NeutralFace",
        "PeekABooGetIn",
        "PeekABooShort",
        "PeekABooSurprised",
        "PeekABooGetOutHappy",
        "PeekABooGetOutSad",
        "ComeHere_AlreadyHere",
        "ComeHere_SearchForFace_FoundFace",
        "MeetCozmoFirstEnrollmentCelebration",
        "HikingReactToNewArea",
        "HikingObserve",
        "HikingSquintStart",
        "HikingSquintEnd",
        "HikingInterestingEdgeThought",
        "HikingLookAround",
        "LookInPlaceForFacesBodyPause",
        "LookInPlaceForFacesHeadMovePause",
        "BuildPyramidFirstBlockUpright",
        "BuildPyramidFirstBlockOnSide",
        "BuildPyramidSecondBlockUpright",
        "BuildPyramidSecondBlockOnSide",
        "BuildPyramidThirdBlockUpright",
        "BuildPyramidThirdBlockOnSide",
        "BuildPyramidSuccess",
        "CantHandleTallStack",
        "KnockOverGrabAttempt",
        "KnockOverEyes",
        "KnockOverSuccess",
        "KnockOverFailure",
        "PutDownBlockPutDown",
        "FeedingSearchRequest",
        "FeedingSearchRequest_Severe",
        "FeedingReactToSeeCube_Normal",
        "FeedingReactToSeeCube_Severe",
        "FeedingSearchFailure",
        "FeedingPlaceLiftOnCube_Normal",
        "FeedingPlaceLiftOnCube_Severe",
        "FeedingReactToShake_Normal",
        "FeedingAteFullEnough_Normal",
        "FeedingAteNotFullEnough_Normal",
        "FeedingAteNotFullEnough_Severe",
        "LaserAcknowledge",
        "LaserPounce",
        "LaserDriveStart",
        "LaserDriveEnd",
        "LaserGetOut",
    )

    def __init__(self):
        super().__init__()
        self.conn = FakeConnection()
        self.animation_groups = {name: None for name in self.ANIMATION_GROUPS}
        self.robot_orientation = pycozmo.robot.RobotOrientation.ON_THREADS
        self.robot_status = 0
        self.played = []
        self.cancelled = 0
        self.wheel_speeds = []
        self.stopped = 0
        self.cliff_stops = []
        self.pose = pycozmo.util.Pose(
            0.0, 0.0, 0.0, angle_z=pycozmo.util.Angle(radians=0.0))
        self.faces = FakeFaces()
        self.cubes = FakeCubes()
        self.camera = []
        self.turns = []
        self.drives = []
        self.head_angles = []

    def play_anim_group(self, name):
        self.played.append(name)

    def cancel_anim(self):
        self.cancelled += 1

    def drive_wheels(self, lwheel_speed, rwheel_speed, *args, **kwargs):
        self.wheel_speeds.append((lwheel_speed, rwheel_speed))

    def stop_all_motors(self):
        self.stopped += 1

    def enable_stop_on_cliff(self, enable=True):
        self.cliff_stops.append(enable)

    def enable_camera(self, enable=True, color=False):
        self.camera.append((enable, color))

    def set_head_angle(self, angle, *args, **kwargs):
        self.head_angles.append(angle)

    def turn_in_place(self, angle, speed=1.0, wait=True, timeout=None):
        self.turns.append(angle.radians)
        heading = self.pose.rotation.angle_z.radians + angle.radians
        self.pose = pycozmo.util.Pose(
            self.pose.position.x, self.pose.position.y, self.pose.position.z,
            angle_z=pycozmo.util.Angle(radians=heading))
        return True

    def drive_straight(self, distance, speed=100.0, wait=True, timeout=None):
        self.drives.append(distance.mm)
        heading = self.pose.rotation.angle_z.radians
        self.pose = pycozmo.util.Pose(
            self.pose.position.x + distance.mm * math.cos(heading),
            self.pose.position.y + distance.mm * math.sin(heading),
            self.pose.position.z,
            angle_z=pycozmo.util.Angle(radians=heading))
        for face in self.faces.visible():
            face.distance = max(0.0, face.distance - distance.mm)
        return True

    def go_to_pose(self, pose, relative_to_robot=False, wait=True, timeout=None):
        self.drives.append("pose")
        self.pose = pose
        return True

    def set_lift_height(self, height, *args, **kwargs):
        self.head_angles.append(("lift", height))


class FakeCubes(list):
    """ Stands in for Cubes, holding a fixed set of cubes. """

    def __init__(self, cubes=()):
        super().__init__(cubes)
        self.carried = None


def make_cube(x=200.0, y=0.0, z=22.5, up_axis=None, connected=True):
    """ A cube the robot knows, upright on the ground unless told otherwise. """
    cube = mock.Mock()
    cube.pose = mock.Mock(x=x, y=y, z=z)
    cube.up_axis = up_axis
    cube.connected = connected
    return cube


class FakeFaces:
    """ Stands in for Faces, holding a fixed set of faces. """

    def __init__(self, faces=()):
        self._faces = list(faces)
        self.enrolled = []

    def available(self):
        return True

    def visible(self):
        return [face for face in self._faces if face.present]

    def enroll(self, name, views=5, timeout=30.0, cancel=None):
        self.enrolled.append(name)
        return True


def make_face(face_id=1, x=500.0, y=0.0, z=0.0, distance=500.0, name=None):
    """ A face in front of the robot, in the tracker's real classes. """
    import time
    face = pycozmo.faces.Face(face_id, time.perf_counter())
    face.present = True
    face.last_seen = time.perf_counter()
    face.pose = pycozmo.faces.FacePose(x=x, y=y, z=z, time=time.perf_counter())
    face.distance = distance
    face.name = name
    return face


class BehaviorTestCase(unittest.TestCase):

    def setUp(self):
        self.cli = FakeClient()

    def make(self, behavior_class, conf=None, behavior_id="TestBehavior"):
        conf = dict(conf or {})
        conf.setdefault("behaviorID", behavior_id)
        behavior = behavior_class(self.cli, conf)
        # Behaviors receive events as a child dispatcher of the client, as activate_behavior does.
        self.cli.add_child_dispatcher(behavior)
        self.addCleanup(self.cli.del_child_dispatcher, behavior)
        return behavior

    def complete_animation(self):
        """ Report the animation the client is playing as finished, the way the controller does. """
        self.cli.dispatch(pycozmo.event.EvtAnimationCompleted, self.cli)

    def posted(self, evt):
        """ The arguments of every posting of an event. """
        return [args for posted, args in self.cli.conn.events if posted is evt]

    def posted_emotion_events(self):
        """ The name of every emotion event the behavior posted. """
        return [args[1] for args in self.posted(pycozmo.event.EvtEmotionEvent)]

    def assertDone(self):
        self.assertTrue(self.posted(pycozmo.event.EvtBehaviorDone))

    def assertNotDone(self):
        self.assertFalse(self.posted(pycozmo.event.EvtBehaviorDone))


class TestBehavior(BehaviorTestCase):

    def test_unimplemented_behavior_completes_immediately(self):
        behavior = self.make(pycozmo.behavior.Behavior)
        behavior.activate()
        self.assertEqual(self.cli.played, [])
        self.assertDone()

    def test_a_behavior_taken_off_the_robot_cannot_report_itself_done(self):
        # A behavior waiting on a timer or an animation can finish just after something else has
        # taken the robot - a reaction, usually. Reporting itself done then would end the reaction
        # instead of itself.
        behavior = self.make(pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["ReactToCliff"]})
        behavior.activate()
        behavior.deactivated = True
        self.complete_animation()
        self.assertNotDone()

    def test_giving_up_is_silenced_the_same_way(self):
        behavior = self.make(pycozmo.behavior.Behavior)
        behavior.deactivated = True
        behavior.activate()
        self.assertNotDone()

    def test_the_client_is_what_marks_a_behavior_off_the_robot(self):
        behavior = self.make(pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["ReactToCliff"]})
        cli = cast(pycozmo.client.Client, self.cli)
        self.assertFalse(behavior.deactivated)
        pycozmo.client.Client.deactivate_behavior(cli, behavior)
        self.assertTrue(behavior.deactivated)
        pycozmo.client.Client.activate_behavior(cli, behavior)
        self.assertFalse(behavior.deactivated, "activating it again puts it back on the robot")


class TestBehaviorPlayAnim(BehaviorTestCase):

    def test_plays_the_configured_trigger(self):
        behavior = self.make(pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["ReactToCliff"]})
        behavior.activate()
        self.assertEqual(self.cli.played, ["ReactToCliff"])
        self.assertNotDone()
        self.complete_animation()
        self.assertDone()

    def test_plays_the_whole_sequence(self):
        # Only the first animation of a sequence used to be played, the behavior reporting itself
        # done as soon as it had finished.
        behavior = self.make(
            pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["ReactToCliff", "ReactToPickup"]})
        behavior.activate()
        self.assertEqual(self.cli.played, ["ReactToCliff"])
        self.complete_animation()
        self.assertEqual(self.cli.played, ["ReactToCliff", "ReactToPickup"])
        self.assertNotDone()
        self.complete_animation()
        self.assertDone()

    def test_unknown_trigger_does_not_leave_the_behavior_active(self):
        # An animation group the assets do not define is never played, so waiting for it to
        # complete would keep the behavior active forever.
        behavior = self.make(pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["NoSuchTrigger"]})
        behavior.activate()
        self.assertEqual(self.cli.played, [])
        self.assertDone()

    def test_unknown_triggers_are_skipped(self):
        behavior = self.make(
            pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["NoSuchTrigger", "ReactToCliff"]})
        behavior.activate()
        self.assertEqual(self.cli.played, ["ReactToCliff"])

    def test_can_be_activated_twice(self):
        # Behaviors are loaded once and activated on every reaction, so an activation must not
        # consume the configured triggers.
        behavior = self.make(pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["ReactToCliff"]})
        behavior.activate()
        self.complete_animation()
        behavior.activate()
        self.assertEqual(self.cli.played, ["ReactToCliff", "ReactToCliff"])

    def test_deactivate_cancels_the_animation(self):
        behavior = self.make(pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["ReactToCliff"]})
        behavior.activate()
        behavior.deactivate()
        self.assertEqual(self.cli.cancelled, 1)


class TestReactions(BehaviorTestCase):
    """ Each reaction plays the animation trigger it is expected to, then reports itself done. """

    REACTIONS = (
        (pycozmo.behavior.BehaviorReactToCliff, None, ["ReactToCliff"]),
        (pycozmo.behavior.BehaviorReactToPickup, None, ["ReactToPickup"]),
        (pycozmo.behavior.BehaviorReactToImpact, None, ["ReactToImpact"]),
        (pycozmo.behavior.BehaviorReactToUnexpectedMovement, None, ["ReactToUnexpectedMovement"]),
        (pycozmo.behavior.BehaviorReactToRobotOnBack, None, ["FlipDownFromBack"]),
        (pycozmo.behavior.BehaviorReactToRobotOnFace, None, ["FacePlantRoll"]),
        (pycozmo.behavior.BehaviorReactToRobotShaken, None, ["DizzyShakeStop"]),
        (pycozmo.behavior.BehaviorReactToPet, None, ["PetDetectionShort"]),
        (pycozmo.behavior.BehaviorReactToCubeMoved, None, ["CubeMovedSense"]),
        (pycozmo.behavior.BehaviorReactToSparked, None, ["SparkGetIn"]),
        (pycozmo.behavior.BehaviorRamIntoBlock, None, ["SoundOnlyRamIntoBlock"]),
        (pycozmo.behavior.BehaviorFistBump, None, ["FistBumpRequestOnce", "FistBumpLeftHanging"]),
        (pycozmo.behavior.BehaviorAcknowledge,
         {"ReactionAnimGroup": "ReactToNewBlockSmall"}, ["ReactToNewBlockSmall"]),
        (pycozmo.behavior.BehaviorAcknowledge,
         {"ReactionAnimGroup": "AcknowledgeObject"}, ["AcknowledgeObject"]),
        (pycozmo.behavior.BehaviorReactToFrustration,
         {"anim": "FrustratedByFailure"}, ["FrustratedByFailure"]),
        (pycozmo.behavior.BehaviorReactToPyramid, None, ["BuildPyramidSuccess"]),
        (pycozmo.behavior.BehaviorReactToStackOfCubes, None, ["AcknowledgeObject"]),
        (pycozmo.behavior.BehaviorPyramidThankYou, None, ["BuildPyramidThankUser"]),
        (pycozmo.behavior.BehaviorEarnedSparks, None, ["EarnedSparks"]),
        (pycozmo.behavior.BehaviorOnConfigSeen,
         {"animTriggers": ["BuildPyramidReactToBase"]}, ["BuildPyramidReactToBase"]),
        (pycozmo.behavior.BehaviorRespondToRenameFace,
         {"animationTrigger": "MeetCozmoRenameFaceSayName"}, ["MeetCozmoRenameFaceSayName"]),
    )

    def test_reactions(self):
        for behavior_class, conf, expected in self.REACTIONS:
            with self.subTest(behavior=behavior_class.__name__, expected=expected):
                self.setUp()
                behavior = self.make(behavior_class, conf, behavior_id=behavior_class.__name__)
                behavior.activate()
                for i, name in enumerate(expected):
                    self.assertEqual(self.cli.played, expected[:i + 1])
                    self.assertNotDone()
                    self.complete_animation()
                self.assertDone()


class TestBehaviorReactToRobotOnSide(BehaviorTestCase):

    def test_left_side(self):
        self.cli.robot_orientation = pycozmo.robot.RobotOrientation.ON_LEFT_SIDE
        behavior = self.make(pycozmo.behavior.BehaviorReactToRobotOnSide)
        behavior.activate()
        self.assertEqual(self.cli.played, ["ReactToOnLeftSide"])
        self.complete_animation()
        self.assertDone()

    def test_right_side(self):
        self.cli.robot_orientation = pycozmo.robot.RobotOrientation.ON_RIGHT_SIDE
        behavior = self.make(pycozmo.behavior.BehaviorReactToRobotOnSide)
        behavior.activate()
        self.assertEqual(self.cli.played, ["ReactToOnRightSide"])
        self.complete_animation()
        self.assertDone()


class TestBehaviorReactToOnCharger(BehaviorTestCase):

    # Short enough to keep the tests quick, long enough not to expire while they run.
    CONF = {
        "behaviorID": "ReactToOnCharger",
        "timeTilSleepAnimation_s": 0.05,
        "timeTilDisconnection_s": 0.1,
    }

    def make_behavior(self, **overrides):
        conf = dict(self.CONF)
        conf.update(overrides)
        return self.make(pycozmo.behavior.BehaviorReactToOnCharger, conf)

    def wait_for_timer(self, behavior):
        """ Wait out the behavior's pending timer. """
        timer = behavior.timer
        self.assertIsNotNone(timer)
        timer.join(2.0)
        self.assertFalse(timer.is_alive())

    def test_reads_the_delays_from_the_configuration(self):
        behavior = self.make_behavior()
        self.assertEqual(behavior.time_til_sleep_animation, 0.05)
        self.assertEqual(behavior.time_til_disconnection, 0.1)

    def test_defaults_the_delays(self):
        behavior = self.make(pycozmo.behavior.BehaviorReactToOnCharger)
        self.assertEqual(behavior.time_til_sleep_animation, 300.0)
        self.assertEqual(behavior.time_til_disconnection, 330.0)

    def test_plays_the_charger_reaction_then_sleeps_then_completes(self):
        behavior = self.make_behavior()
        behavior.activate()
        self.assertEqual(self.cli.played, ["PlacedOnCharger"])

        # The reaction being over only starts the wait before sleep.
        self.complete_animation()
        self.assertNotDone()
        self.wait_for_timer(behavior)
        self.assertEqual(self.cli.played, ["PlacedOnCharger", "GoToSleepGetIn"])

        self.complete_animation()
        self.assertEqual(
            self.cli.played, ["PlacedOnCharger", "GoToSleepGetIn", "GoToSleepSleeping"])
        self.assertNotDone()

        # Only the disconnection delay, after the sleep animation, ends the behavior.
        self.complete_animation()
        self.assertNotDone()
        self.wait_for_timer(behavior)
        self.assertDone()

    def test_completes_when_taken_off_the_charger(self):
        behavior = self.make_behavior(timeTilSleepAnimation_s=300.0, timeTilDisconnection_s=330.0)
        behavior.activate()
        self.complete_animation()
        self.assertNotDone()
        self.cli.dispatch(pycozmo.event.EvtRobotOnChargerChange, self.cli, False)
        self.assertDone()
        self.assertIsNone(behavior.timer)

    def test_stays_active_while_on_the_charger(self):
        behavior = self.make_behavior(timeTilSleepAnimation_s=300.0, timeTilDisconnection_s=330.0)
        behavior.activate()
        self.complete_animation()
        self.cli.dispatch(pycozmo.event.EvtRobotOnChargerChange, self.cli, True)
        self.assertNotDone()

    def test_deactivate_cancels_the_pending_timer(self):
        behavior = self.make_behavior(timeTilSleepAnimation_s=300.0, timeTilDisconnection_s=330.0)
        behavior.activate()
        self.complete_animation()
        self.assertIsNotNone(behavior.timer)
        behavior.deactivate()
        self.assertIsNone(behavior.timer)
        self.assertEqual(self.cli.cancelled, 1)


class TestWantsToRun(BehaviorTestCase):
    """
    What a behavior answers when the activity engine offers it the robot.

    Nothing consults this for a reaction: a trigger names its behavior and the brain runs it. It is
    the activity engine, choosing what to do when nothing has happened, that asks.
    """

    def test_an_unimplemented_behavior_never_runs(self):
        # Activating one only logs that and reports it done, and the engine would offer it the
        # robot again at once.
        self.assertFalse(self.make(pycozmo.behavior.Behavior).wants_to_run())

    def test_an_animation_behavior_runs_when_its_animation_is_there(self):
        behavior = self.make(pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["ReactToCliff"]})
        self.assertTrue(behavior.wants_to_run())

    def test_an_animation_behavior_holds_back_without_its_animation(self):
        behavior = self.make(pycozmo.behavior.BehaviorPlayAnim, {"animTriggers": ["NoSuchTrigger"]})
        self.assertFalse(behavior.wants_to_run())

    def test_a_strategy_this_library_cannot_read_holds_a_behavior_back(self):
        # ReactToObstacle is the only behavior in the resources carrying one, and it asks for
        # ObstacleDetected. Nothing here sees an obstacle.
        behavior = self.make(pycozmo.behavior.BehaviorPlayAnim, {
            "animTriggers": ["ReactToCliff"],
            "wantsToRunStrategyConfig": {"strategyType": "ObstacleDetected"}})
        self.assertFalse(behavior.wants_to_run())


class TestBehaviorDriveOffCharger(BehaviorTestCase):

    def make_behavior(self, extra_distance=60.0):
        return self.make(pycozmo.behavior.BehaviorDriveOffCharger,
                         {"extraDistanceToDrive_mm": extra_distance}, "DriveOffCharger")

    def test_it_only_runs_on_the_charger(self):
        behavior = self.make_behavior()
        self.assertFalse(behavior.wants_to_run())
        self.cli.robot_status = pycozmo.robot.RobotStatusFlag.IS_ON_CHARGER
        self.assertTrue(behavior.wants_to_run())

    def test_it_drives_forward(self):
        # It used to report itself done without moving, so a robot on its charger stayed on it -
        # and, offered the robot by the engine, would have been offered it again straight away.
        behavior = self.make_behavior()
        behavior.activate()
        self.assertEqual(behavior.timer.interval, (40.0 + 60.0) / behavior.SPEED)
        self.assertEqual(self.cli.wheel_speeds, [(behavior.SPEED, behavior.SPEED)])
        self.assertEqual(self.posted_emotion_events(), ["DriveOffCharger"])
        self.assertNotDone()

    def test_it_does_not_stop_at_the_chargers_lip(self):
        # The robot takes the lip of its charger for a cliff, and with its cliff stop on, as the brain has it, it
        # came off by 24 mm and no more. The stop is back on once it has gone far enough, or been told to stop.
        behavior = self.make_behavior()
        behavior.activate()
        self.assertEqual(self.cli.cliff_stops, [False])
        behavior.timer.cancel()
        behavior._arrived()
        self.assertEqual(self.cli.cliff_stops, [False, True])
        behavior.activate()
        behavior.deactivate()
        self.assertEqual(self.cli.cliff_stops, [False, True, False, True])

    def test_it_stops_where_the_configuration_says(self):
        behavior = self.make_behavior(extra_distance=0.0)
        behavior.activate()
        self.assertEqual(behavior.timer.interval, 40.0 / behavior.SPEED)

    def test_it_stops_when_it_has_gone_far_enough(self):
        behavior = self.make_behavior()
        behavior.activate()
        behavior.timer.cancel()
        behavior._arrived()
        self.assertEqual(self.cli.stopped, 1)
        self.assertDone()

    def test_it_does_not_set_off_again_at_once(self):
        # The robot's status takes a moment to show the charger clear, and a second attempt in the
        # meantime would drive it twice as far as the configuration asks.
        behavior = self.make_behavior()
        self.cli.robot_status = pycozmo.robot.RobotStatusFlag.IS_ON_CHARGER
        behavior.activate()
        behavior.timer.cancel()
        self.assertFalse(behavior.wants_to_run())

    def test_it_gives_up_rather_than_driving_across_the_table(self):
        # A status stuck on the charger would otherwise have the robot drive off it for ever.
        behavior = self.make_behavior()
        self.cli.robot_status = pycozmo.robot.RobotStatusFlag.IS_ON_CHARGER
        for attempt in range(behavior.MAX_ATTEMPTS):
            behavior.last_run = 0.0
            self.assertTrue(behavior.wants_to_run(), attempt)
            behavior.activate()
            behavior.timer.cancel()
        behavior.last_run = 0.0
        self.assertFalse(behavior.wants_to_run())
        # Taken off the charger, it is willing again.
        self.cli.robot_status = 0
        self.assertFalse(behavior.wants_to_run())
        self.cli.robot_status = pycozmo.robot.RobotStatusFlag.IS_ON_CHARGER
        self.assertTrue(behavior.wants_to_run())

    def test_being_deactivated_stops_the_motors(self):
        behavior = self.make_behavior()
        behavior.activate()
        behavior.deactivate()
        self.assertIsNone(behavior.timer)
        self.assertEqual(self.cli.stopped, 1)


class TestGetBehaviorClassFromDict(unittest.TestCase):
    """ Every reaction behavior of reactionTrigger_behavior_map.json that has an animation. """

    EXPECTED = {
        "AcknowledgeFace": pycozmo.behavior.BehaviorAcknowledge,
        "AcknowledgeObject": pycozmo.behavior.BehaviorAcknowledge,
        "DriveOffCharger": pycozmo.behavior.BehaviorDriveOffCharger,
        "FistBump": pycozmo.behavior.BehaviorFistBump,
        "PlayAnim": pycozmo.behavior.BehaviorPlayAnim,
        "PlayArbitraryAnim": pycozmo.behavior.BehaviorPlayArbitraryAnim,
        "RamIntoBlock": pycozmo.behavior.BehaviorRamIntoBlock,
        "ReactToCliff": pycozmo.behavior.BehaviorReactToCliff,
        "ReactToCubeMoved": pycozmo.behavior.BehaviorReactToCubeMoved,
        "ReactToFrustration": pycozmo.behavior.BehaviorReactToFrustration,
        "ReactToImpact": pycozmo.behavior.BehaviorReactToImpact,
        "ReactToOnCharger": pycozmo.behavior.BehaviorReactToOnCharger,
        "ReactToPet": pycozmo.behavior.BehaviorReactToPet,
        "ReactToPickup": pycozmo.behavior.BehaviorReactToPickup,
        "ReactToRobotOnBack": pycozmo.behavior.BehaviorReactToRobotOnBack,
        "ReactToRobotOnFace": pycozmo.behavior.BehaviorReactToRobotOnFace,
        "ReactToRobotOnSide": pycozmo.behavior.BehaviorReactToRobotOnSide,
        "ReactToRobotShaken": pycozmo.behavior.BehaviorReactToRobotShaken,
        "ReactToSparked": pycozmo.behavior.BehaviorReactToSparked,
        "ReactToUnexpectedMovement": pycozmo.behavior.BehaviorReactToUnexpectedMovement,
        "ReactToPyramid": pycozmo.behavior.BehaviorReactToPyramid,
        "ReactToStackOfCubes": pycozmo.behavior.BehaviorReactToStackOfCubes,
        "ReactToReturnedToTreads": pycozmo.behavior.BehaviorReactToReturnedToTreads,
        "ReactToPlacedOnSlope": pycozmo.behavior.BehaviorReactToPlacedOnSlope,
        "ReactToMotorCalibration": pycozmo.behavior.BehaviorReactToMotorCalibration,
        "OnConfigSeen": pycozmo.behavior.BehaviorOnConfigSeen,
        "PyramidThankYou": pycozmo.behavior.BehaviorPyramidThankYou,
        "EarnedSparks": pycozmo.behavior.BehaviorEarnedSparks,
        "RespondToRenameFace": pycozmo.behavior.BehaviorRespondToRenameFace,
        "PlayAnimWithFace": pycozmo.behavior.BehaviorPlayAnimWithFace,
    }

    def test_known_classes(self):
        for behavior_class, expected in self.EXPECTED.items():
            with self.subTest(behavior_class=behavior_class):
                self.assertIs(
                    pycozmo.behavior.get_behavior_class_from_dict({"behaviorClass": behavior_class}),
                    expected)

    def test_unknown_class_falls_back(self):
        # Classes the resources name but nothing implements yet, and names nothing does.
        for behavior_class in ("Bouncer", "GuardDog", "NoSuchBehaviorClass"):
            with self.subTest(behavior_class=behavior_class):
                self.assertIs(
                    pycozmo.behavior.get_behavior_class_from_dict({"behaviorClass": behavior_class}),
                    pycozmo.behavior.Behavior)


def cozmo_assets_available():
    try:
        pycozmo.util.check_assets()
    except pycozmo.exception.ResourcesNotFound:
        return False
    return True


@unittest.skipUnless(cozmo_assets_available(), "Cozmo assets not downloaded.")
class TestAgainstCozmoAssets(unittest.TestCase):
    """ Check the behaviors against the Anki resources they name animations from. """

    resource_dir: str
    anim_triggers: Set[str]
    behaviors: Dict[str, pycozmo.behavior.Behavior]
    reaction_triggers: Dict[str, List[pycozmo.behavior.ReactionTrigger]]

    @classmethod
    def setUpClass(cls):
        cls.resource_dir = str(pycozmo.util.get_cozmo_asset_dir())
        # Animation triggers are the "CladEvent" names of the animation trigger map, and are only
        # usable if the animation group they name is on disk, since the loader skips the others.
        trigger_map = pycozmo.json_loader.load_json_file(os.path.join(
            cls.resource_dir, "cozmo_resources", "assets", "animationGroupMaps",
            "AnimationTriggerMap.json"))
        groups = set()
        for _, _, files in os.walk(os.path.join(
                cls.resource_dir, "cozmo_resources", "assets", "animationGroups")):
            groups.update(files)
        cls.anim_triggers = {pair["CladEvent"] for pair in trigger_map["Pairs"]
                             if pair["AnimName"] + ".json" in groups}
        cls.behaviors = pycozmo.behavior.load_behaviors(cls.resource_dir, cast(pycozmo.client.Client, FakeClient()))
        cls.reaction_triggers = pycozmo.behavior.load_reaction_trigger_behavior_map(cls.resource_dir)

    def test_every_named_animation_trigger_exists(self):
        # A trigger the resources do not define is dropped on activation, so a behavior whose name
        # for one is wrong would silently do nothing at all.
        for behavior in self.behaviors.values():
            if not isinstance(behavior, pycozmo.behavior.BehaviorPlayAnim):
                continue
            for trigger in behavior.get_anim_triggers():
                with self.subTest(behavior=behavior.get_id(), trigger=trigger):
                    self.assertIn(trigger, self.anim_triggers)

    def test_reactions_resolve_to_an_animation(self):
        # The reactions that are expected to be implemented, and the animation each plays.
        expected = {
            "AcknowledgeFace": ["ReactToNewBlockSmall"],
            "AcknowledgeObject": ["AcknowledgeObject"],
            "FistBump": ["FistBumpRequestOnce", "FistBumpLeftHanging"],
            "Hiccup": ["Hiccup"],
            "RamIntoBlock": ["SoundOnlyRamIntoBlock"],
            "ReactToCliff": ["ReactToCliff"],
            "ReactToCubeMoved": ["CubeMovedSense"],
            "ReactToFrustrationMajor": ["FrustratedByFailureMajor"],
            "ReactToFrustrationMinor": ["FrustratedByFailure"],
            "ReactToImpact": ["ReactToImpact"],
            "ReactToOnCharger": ["PlacedOnCharger"],
            "ReactToPet": ["PetDetectionShort"],
            "ReactToPickup": ["ReactToPickup"],
            "ReactToRobotOnBack": ["FlipDownFromBack"],
            "ReactToRobotOnFace": ["FacePlantRoll"],
            "ReactToRobotOnSide": ["ReactToOnRightSide"],
            "ReactToRobotShaken": ["DizzyShakeStop"],
            "ReactToSparked": ["SparkGetIn"],
            "ReactToUnexpectedMovement": ["ReactToUnexpectedMovement"],
        }
        for behavior_id, triggers in expected.items():
            with self.subTest(behavior=behavior_id):
                behavior = self.behaviors[behavior_id]
                assert isinstance(behavior, pycozmo.behavior.BehaviorPlayAnim)
                self.assertEqual(list(behavior.get_anim_triggers()), triggers)
                for trigger in triggers:
                    self.assertIn(trigger, self.anim_triggers)

    def test_quiet_reactions_are_implemented(self):
        # AnimationTriggerMap.json holds no animation for these, under their behaviour ID, their
        # reaction trigger, or any name close to either, so they are quiet behaviors rather than
        # animation players — but they are implemented classes all the same.
        expected = {
            "ReactToMotorCalibration": pycozmo.behavior.BehaviorReactToMotorCalibration,
            "ReactToPlacedOnSlope": pycozmo.behavior.BehaviorReactToPlacedOnSlope,
            "ReactToReturnedToTreads": pycozmo.behavior.BehaviorReactToReturnedToTreads,
        }
        for behavior_id, cls in expected.items():
            with self.subTest(behavior=behavior_id):
                self.assertIs(type(self.behaviors[behavior_id]), cls)

    def test_every_behavior_class_is_implemented(self):
        # Every behaviorClass in the resources resolves to a real class: the engine would
        # otherwise run a silent no-op base Behavior. EXPECTED_UNMAPPED names the ones still
        # to do; shrink it as waves of behaviors land.
        from pycozmo.behavior import get_behavior_class_from_dict
        classes = {behavior.conf["behaviorClass"] for behavior in self.behaviors.values()}
        unmapped = {c for c in classes
                    if get_behavior_class_from_dict({"behaviorClass": c}) is pycozmo.behavior.Behavior}
        self.assertEqual(unmapped, self.EXPECTED_UNMAPPED)

    #: Behavior classes in the resources with no implementation yet.
    EXPECTED_UNMAPPED = frozenset({
        "Bouncer",
        "DevTurnInPlaceTest",
        "DockingTestSimple",
        "FactoryCentroidExtractor",
        "FactoryTest",
        "FireTruckAlarm",
        "GuardDog",
        "LiftLoadTest",
        "OnboardingShowCube",
    })

    def test_every_reaction_trigger_has_a_behavior(self):
        for reactions in self.reaction_triggers.values():
            for reaction in reactions:
                with self.subTest(reaction=reaction.name, behavior=reaction.behavior_id):
                    self.assertIn(reaction.behavior_id, self.behaviors)

    def test_frustration_keeps_both_of_its_behaviors(self):
        # The map is keyed by trigger and Frustration appears twice, so keeping one behavior per
        # trigger dropped the minor variant and always ran the major one.
        frustration = self.reaction_triggers["Frustration"]
        self.assertEqual({reaction.behavior_id for reaction in frustration},
                         {"ReactToFrustrationMinor", "ReactToFrustrationMajor"})
        by_id = {reaction.behavior_id: reaction for reaction in frustration}
        self.assertEqual(by_id["ReactToFrustrationMinor"].max_confidence, -0.6)
        self.assertEqual(by_id["ReactToFrustrationMajor"].max_confidence, -0.9)
        self.assertEqual(by_id["ReactToFrustrationMinor"].cooldown_time, 60.0)


class TestBehaviorQuietReactions(BehaviorTestCase):
    """ Reactions with no animation finish at once, playing nothing. """

    def test_returned_to_treads(self):
        behavior = self.make(pycozmo.behavior.BehaviorReactToReturnedToTreads)
        behavior.activate()
        self.assertEqual(self.cli.played, [])
        self.assertDone()

    def test_motor_calibration(self):
        behavior = self.make(pycozmo.behavior.BehaviorReactToMotorCalibration)
        behavior.activate()
        self.assertEqual(self.cli.played, [])
        self.assertDone()


class TestBehaviorReactToPlacedOnSlope(BehaviorTestCase):
    """ Backing off a slope drives both wheels back, then stops and finishes. """

    def test_backs_up_then_finishes(self):
        behavior = self.make(pycozmo.behavior.BehaviorReactToPlacedOnSlope)
        behavior.activate()
        self.assertEqual(self.cli.wheel_speeds, [(-40.0, -40.0)])
        self.assertNotDone()
        timer = behavior.timer
        self.assertIsNotNone(timer)
        assert timer is not None
        timer.join(3.0)
        self.assertFalse(timer.is_alive())
        self.assertEqual(self.cli.stopped, 1)
        self.assertDone()

    def test_deactivate_stops_the_wheels(self):
        behavior = self.make(pycozmo.behavior.BehaviorReactToPlacedOnSlope)
        behavior.activate()
        behavior.deactivate()
        self.assertIsNone(behavior.timer)
        self.assertEqual(self.cli.stopped, 1)


class StubNeeds:
    """ Stands in for Needs, answering one bracket per need name. """

    def __init__(self, brackets):
        self.brackets = dict(brackets)

    def bracket(self, name):
        return self.brackets.get(name, "Full")


class TestBehaviorPlayAnimWithFace(BehaviorTestCase):
    """ The needs-level variants pick their animation by bracket; the rest play their triggers. """

    def test_explicit_triggers_play_unchanged(self):
        behavior = self.make(
            pycozmo.behavior.BehaviorPlayAnimWithFace,
            {"animTriggers": ["NeedsMildLowEnergyRequest"]},
            behavior_id="FeedingPlayRequestAtFace")
        behavior.activate()
        self.assertEqual(self.cli.played, ["NeedsMildLowEnergyRequest"])

    def test_critical_plays_the_severe_request(self):
        behavior = self.make(
            pycozmo.behavior.BehaviorPlayAnimWithFace,
            {"animTriggers": ["NeutralFace"]},
            behavior_id="VC_HowAreYouDoing_Energy")
        behavior.needs = StubNeeds({"Energy": "Critical"})
        behavior.activate()
        self.assertEqual(self.cli.played, ["NeedsSevereLowEnergyRequest"])

    def test_warning_plays_the_mild_request(self):
        behavior = self.make(
            pycozmo.behavior.BehaviorPlayAnimWithFace,
            {"animTriggers": ["NeutralFace"]},
            behavior_id="VC_HowAreYouDoing_Play")
        behavior.needs = StubNeeds({"Play": "Warning"})
        behavior.activate()
        self.assertEqual(self.cli.played, ["NeedsMildLowPlayRequest"])

    def test_full_plays_the_neutral_face(self):
        behavior = self.make(
            pycozmo.behavior.BehaviorPlayAnimWithFace,
            {"animTriggers": ["NeutralFace"]},
            behavior_id="VC_HowAreYouDoing_Repair")
        behavior.needs = StubNeeds({"Repair": "Full"})
        behavior.activate()
        self.assertEqual(self.cli.played, ["NeutralFace"])

    def test_without_needs_plays_the_neutral_face(self):
        behavior = self.make(
            pycozmo.behavior.BehaviorPlayAnimWithFace,
            {"animTriggers": ["NeutralFace"]},
            behavior_id="VC_HowAreYouDoing_Energy")
        behavior.activate()
        self.assertEqual(self.cli.played, ["NeutralFace"])


def complete_until_done(testcase, cli, timeout=15.0):
    """ Keep completing animations until the behavior reports done or time runs out. """
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if testcase.posted(pycozmo.event.EvtBehaviorDone):
            return True
        cli.dispatch(pycozmo.event.EvtAnimationCompleted, cli)
        time.sleep(0.05)
    return bool(testcase.posted(pycozmo.event.EvtBehaviorDone))


class TestBehaviorFindFaces(BehaviorTestCase):
    """ A face ahead ends the search at once; otherwise the robot looks all round. """

    def test_face_ahead_finishes_at_once(self):
        self.cli.faces = FakeFaces([make_face()])
        behavior = self.make(pycozmo.face_behaviors.BehaviorFindFaces)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertEqual(self.cli.turns, [])

    def test_no_face_turns_all_the_way_round(self):
        behavior = self.make(pycozmo.face_behaviors.BehaviorFindFaces)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=10.0))
        self.assertEqual(len(self.cli.turns), 12)

    def test_search_for_face_behaves_the_same(self):
        self.cli.faces = FakeFaces([make_face()])
        behavior = self.make(pycozmo.face_behaviors.BehaviorSearchForFace)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertEqual(self.cli.turns, [])


class TestBehaviorDriveToFace(BehaviorTestCase):
    """ Drives to a face in legs, stopping at a social distance. """

    def test_already_there_says_so(self):
        self.cli.faces = FakeFaces([make_face(distance=100.0)])
        behavior = self.make(pycozmo.face_behaviors.BehaviorDriveToFace)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli))
        self.assertEqual(self.cli.drives, [])
        self.assertEqual(self.cli.played, ["ComeHere_AlreadyHere"])

    def test_drives_over_in_legs(self):
        self.cli.faces = FakeFaces([make_face(x=500.0, distance=500.0)])
        behavior = self.make(pycozmo.face_behaviors.BehaviorDriveToFace)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli))
        self.assertEqual(self.cli.drives, [100.0, 100.0, 50.0])
        self.assertEqual(self.cli.played, ["ComeHere_SearchForFace_FoundFace"])

    def test_no_face_gives_up(self):
        behavior = self.make(pycozmo.face_behaviors.BehaviorDriveToFace)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertEqual(self.cli.drives, [])
        self.assertEqual(self.cli.played, [])


class TestBehaviorEnrollFace(BehaviorTestCase):
    """ Enrolls a lone face; more than one is refused. """

    def test_enrolls_a_single_face(self):
        self.cli.faces = FakeFaces([make_face()])
        behavior = self.make(pycozmo.face_behaviors.BehaviorEnrollFace)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli))
        self.assertEqual(self.cli.faces.enrolled, ["Friend"])
        self.assertEqual(self.cli.played, ["MeetCozmoFirstEnrollmentCelebration"])

    def test_too_many_faces_refuses(self):
        self.cli.faces = FakeFaces([make_face(face_id=1), make_face(face_id=2)])
        behavior = self.make(pycozmo.face_behaviors.BehaviorEnrollFace,
                             {"tooManyFacesTimeout_sec": 0.2})
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=5.0))
        self.assertEqual(self.cli.faces.enrolled, [])
        self.assertEqual(self.cli.played, [])


class TestBehaviorInteractWithFaces(BehaviorTestCase):
    """ Follows the nearest face, turning towards its bearing. """

    def test_turns_towards_the_face(self):
        self.cli.faces = FakeFaces([make_face(x=500.0, y=500.0, distance=700.0)])
        behavior = self.make(
            pycozmo.face_behaviors.BehaviorInteractWithFaces,
            {"params": {"minTimeToTrackFace_s": 0.1, "maxTimeToTrackFace_s": 0.4,
                        "minClampPeriod_s": 0.05, "maxClampPeriod_s": 0.05}})
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=5.0))
        self.assertTrue(self.cli.turns)
        self.assertAlmostEqual(self.cli.turns[0], math.pi / 4.0, places=5)

    def test_no_face_leaves_at_once(self):
        behavior = self.make(
            pycozmo.face_behaviors.BehaviorInteractWithFaces,
            {"params": {"minTimeToTrackFace_s": 0.1, "maxTimeToTrackFace_s": 0.4,
                        "minClampPeriod_s": 0.05, "maxClampPeriod_s": 0.05}})
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertEqual(self.cli.turns, [])


class TestBehaviorPeekABoo(BehaviorTestCase):
    """ Three rounds with a face watching, one sad round without. """

    def test_plays_three_rounds_with_a_face(self):
        self.cli.faces = FakeFaces([make_face()])
        behavior = self.make(pycozmo.face_behaviors.BehaviorPeekABoo)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=20.0))
        self.assertEqual(
            self.cli.played,
            ["PeekABooGetIn",
             "PeekABooShort", "PeekABooSurprised",
             "PeekABooShort", "PeekABooSurprised",
             "PeekABooShort", "PeekABooSurprised",
             "PeekABooGetOutHappy"])

    def test_no_face_gets_one_sad_round(self):
        behavior = self.make(pycozmo.face_behaviors.BehaviorPeekABoo)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=20.0))
        self.assertEqual(
            self.cli.played,
            ["PeekABooGetIn", "PeekABooShort", "PeekABooSurprised", "PeekABooGetOutSad"])


class TestBehaviorLookForFaceAndCube(BehaviorTestCase):
    """ Turns towards a named face; otherwise sweeps and finishes. """

    def test_named_face_turns_towards_it(self):
        self.cli.faces = FakeFaces([make_face(x=0.0, y=500.0, name="Aunt")])
        behavior = self.make(pycozmo.face_behaviors.BehaviorLookForFaceAndCube)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=10.0))
        self.assertTrue(self.cli.turns)
        # A sweep step comes first, then the remaining bearing to the face.
        self.assertAlmostEqual(sum(self.cli.turns), math.pi / 2.0, places=1)

    def test_nothing_found_sweeps_then_finishes(self):
        behavior = self.make(pycozmo.face_behaviors.BehaviorLookForFaceAndCube)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=10.0))
        self.assertEqual(len(self.cli.turns), 4)


class TestBehaviorThinkAboutBeacons(BehaviorTestCase):
    """ New ground is remembered as a beacon area, with the new-area animation. """

    CONF = {"params": {"beaconRadius_mm": 175.0, "newAreaAnimTrigger": "HikingReactToNewArea"}}

    def test_new_ground_is_remembered(self):
        from pycozmo.explore_behaviors import beacons_for
        behavior = self.make(pycozmo.explore_behaviors.BehaviorThinkAboutBeacons, self.CONF)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli))
        areas = beacons_for(self.cli).areas
        self.assertEqual(len(areas), 1)
        self.assertEqual(self.cli.played, ["HikingReactToNewArea"])

    def test_known_ground_is_quiet(self):
        from pycozmo.explore_behaviors import beacons_for
        beacons_for(self.cli).areas.append((0.0, 0.0, 175.0))
        behavior = self.make(pycozmo.explore_behaviors.BehaviorThinkAboutBeacons, self.CONF)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertEqual(self.cli.played, [])


class TestBehaviorBringCubeToBeacon(BehaviorTestCase):
    """ Carried cubes ride to the nearest beacon; others are picked up first. """

    def seed_beacon(self):
        from pycozmo.explore_behaviors import beacons_for
        beacons_for(self.cli).areas.append((300.0, 0.0, 175.0))

    def test_carried_cube_rides_to_the_beacon(self):
        self.cli.cubes = mock.Mock(carried=mock.Mock())
        self.seed_beacon()
        behavior = self.make(pycozmo.explore_behaviors.BehaviorBringCubeToBeacon)
        with mock.patch("pycozmo.cube_handling.put_down_cube", return_value=True) as put_down:
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli))
        put_down.assert_called_once()
        self.assertIn("pose", self.cli.drives)

    def test_loose_cube_is_picked_up_first(self):
        self.cli.cubes = mock.Mock(carried=None)
        self.seed_beacon()
        behavior = self.make(pycozmo.explore_behaviors.BehaviorBringCubeToBeacon)
        with mock.patch("pycozmo.explore_behaviors.usable_cubes", return_value=[mock.Mock()]), \
                mock.patch("pycozmo.cube_handling.pick_up_cube", return_value=True) as pick_up, \
                mock.patch("pycozmo.cube_handling.put_down_cube", return_value=True):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli))
        pick_up.assert_called_once()

    def test_no_beacon_gives_up(self):
        self.cli.cubes = mock.Mock(carried=mock.Mock())
        behavior = self.make(pycozmo.explore_behaviors.BehaviorBringCubeToBeacon)
        with mock.patch("pycozmo.cube_handling.put_down_cube") as put_down:
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        put_down.assert_not_called()
        self.assertNotIn("pose", self.cli.drives)

    def test_recent_failure_holds_off(self):
        behavior = self.make(pycozmo.explore_behaviors.BehaviorBringCubeToBeacon)
        behavior.last_failure = time.perf_counter()
        self.assertFalse(behavior.wants_to_run())


class TestBehaviorVisitInterestingEdge(BehaviorTestCase):
    """ Squints, edges closer, and has a proper look. """

    def test_looks_at_the_edge(self):
        behavior = self.make(pycozmo.explore_behaviors.BehaviorVisitInterestingEdge)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli))
        self.assertEqual(self.cli.played,
                         ["HikingSquintStart", "HikingObserve", "HikingSquintEnd"])
        self.assertTrue(self.cli.drives)

    def test_blocked_drive_means_no_edge(self):
        behavior = self.make(
            pycozmo.explore_behaviors.BehaviorVisitInterestingEdge,
            {"params": {"edgesNotFoundAnimTrigger": "HikingInterestingEdgeThought",
                        "squintStartAnimTrigger": "HikingSquintStart"}})
        with mock.patch.object(self.cli, "drive_straight", return_value=False):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli))
        self.assertEqual(self.cli.played,
                         ["HikingSquintStart", "HikingInterestingEdgeThought"])


class TestBehaviorExploreVisitPossibleMarker(BehaviorTestCase):
    """ Drives to within looking distance of something half-seen. """

    def test_drives_out_to_a_cube(self):
        cube = mock.Mock()
        cube.pose = mock.Mock(x=500.0, y=0.0)
        behavior = self.make(pycozmo.explore_behaviors.BehaviorExploreVisitPossibleMarker)
        with mock.patch("pycozmo.explore_behaviors.usable_cubes", return_value=[cube]):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=5.0))
        self.assertIn("pose", self.cli.drives)

    def test_close_enough_stays_put(self):
        cube = mock.Mock()
        cube.pose = mock.Mock(x=100.0, y=0.0)
        behavior = self.make(pycozmo.explore_behaviors.BehaviorExploreVisitPossibleMarker)
        with mock.patch("pycozmo.explore_behaviors.usable_cubes", return_value=[cube]):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertNotIn("pose", self.cli.drives)

    def test_nothing_known_gives_up(self):
        behavior = self.make(pycozmo.explore_behaviors.BehaviorExploreVisitPossibleMarker)
        with mock.patch("pycozmo.explore_behaviors.usable_cubes", return_value=[]):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertNotIn("pose", self.cli.drives)


class TestBehaviorLookAround(BehaviorTestCase):
    """ The staged scan turns right round and pauses on the wait trigger. """

    CONF = {"params": {
        "behavior_NumberOfScansBeforeStop": 1,
        "behavior_CanCarryCube": True,
        "s0_MainTurnCWChance": 1.0,
        "sx_BodyTurnSpeed_degPerSec": 120.0,
        "s1_BodyAngleRangeMin_deg": 30.0,
        "s1_BodyAngleRangeMax_deg": 30.0,
        "s1_HeadAngleRangeMin_deg": 0.0,
        "s1_HeadAngleRangeMax_deg": 0.0,
        "s2_WaitMin_sec": 0.01,
        "s2_WaitMax_sec": 0.01,
        "s2_WaitAnimTrigger": "LookInPlaceForFacesBodyPause",
        "s4_HeadAngleChangesMin": 1,
        "s4_HeadAngleChangesMax": 1,
        "s4_WaitBetweenChangesMin_sec": 0.01,
        "s4_WaitBetweenChangesMax_sec": 0.01,
        "s4_WaitAnimTrigger": "LookInPlaceForFacesHeadMovePause",
    }}

    def test_scan_turns_full_circle(self):
        behavior = self.make(pycozmo.explore_behaviors.BehaviorExploreLookAroundInPlace, self.CONF)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=15.0))
        self.assertEqual(len(self.cli.turns), 12)
        self.assertEqual(self.cli.camera, [(True, False)])
        self.assertIn("LookInPlaceForFacesBodyPause", self.cli.played)

    def test_memory_map_scan_uses_its_own_params(self):
        behavior = self.make(pycozmo.explore_behaviors.BehaviorLookInPlaceMemoryMap, self.CONF)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=15.0))
        self.assertEqual(len(self.cli.turns), 12)


class TestBehaviorBuildPyramidBase(BehaviorTestCase):
    """ Lays two blocks side by side, reacting happy to each upright one. """

    def test_lays_two_blocks(self):
        self.cli.cubes = FakeCubes([make_cube(x=200.0), make_cube(x=400.0), make_cube(x=600.0)])
        behavior = self.make(pycozmo.cube_behaviors.BehaviorBuildPyramidBase)
        with mock.patch("pycozmo.cube_handling.pick_up_cube", return_value=True), \
                mock.patch("pycozmo.cube_handling.put_down_cube", return_value=True):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=20.0))
        self.assertEqual(self.cli.played,
                         ["BuildPyramidFirstBlockUpright", "BuildPyramidSecondBlockUpright"])


class TestBehaviorBuildPyramid(BehaviorTestCase):
    """ Lays the base, lifts a third block on top, and celebrates. """

    def test_builds_all_the_way_up(self):
        self.cli.cubes = FakeCubes([make_cube(x=200.0), make_cube(x=400.0), make_cube(x=600.0)])
        behavior = self.make(pycozmo.cube_behaviors.BehaviorBuildPyramid)
        with mock.patch("pycozmo.cube_handling.pick_up_cube", return_value=True), \
                mock.patch("pycozmo.cube_handling.put_down_cube", return_value=True), \
                mock.patch("pycozmo.cube_handling.place_on_cube", return_value=True):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=20.0))
        self.assertEqual(self.cli.played,
                         ["BuildPyramidFirstBlockUpright", "BuildPyramidSecondBlockUpright",
                          "BuildPyramidSuccess"])


class TestBehaviorCantHandleTallStack(BehaviorTestCase):
    """ Looks a tall stack over, sadly - unless it moves first. """

    def stack(self):
        self.cli.cubes = FakeCubes([make_cube(x=200.0, z=22.5),
                                    make_cube(x=200.0, z=67.5),
                                    make_cube(x=200.0, z=112.5)])

    def test_looks_then_sighs(self):
        self.stack()
        behavior = self.make(pycozmo.cube_behaviors.BehaviorCantHandleTallStack)
        self.assertTrue(behavior.wants_to_run())
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=20.0))
        self.assertEqual(self.cli.played, ["CantHandleTallStack"])
        self.assertTrue(self.cli.head_angles)

    def test_a_stack_coming_down_is_left_alone(self):
        self.stack()
        behavior = self.make(pycozmo.cube_behaviors.BehaviorCantHandleTallStack)
        behavior.activate()
        time.sleep(2.2)
        self.cli.cubes[2].pose = mock.Mock(x=260.0, y=0.0, z=22.5)
        self.assertTrue(complete_until_done(self, self.cli, timeout=20.0))
        self.assertEqual(self.cli.played, [])

    def test_no_stack_nothing_to_handle(self):
        behavior = self.make(pycozmo.cube_behaviors.BehaviorCantHandleTallStack)
        self.assertFalse(behavior.wants_to_run())


class TestBehaviorCheckForStackAtInterval(BehaviorTestCase):
    """ A stack found gets acknowledged; nothing found ends quietly. """

    def test_stack_found_gets_acknowledged(self):
        self.cli.cubes = FakeCubes([make_cube(x=200.0, z=22.5),
                                    make_cube(x=200.0, z=67.5),
                                    make_cube(x=200.0, z=112.5)])
        behavior = self.make(pycozmo.cube_behaviors.BehaviorCheckForStackAtInterval)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli))
        self.assertEqual(self.cli.played, ["AcknowledgeObject"])

    def test_no_stack_ends_quietly(self):
        self.cli.cubes = FakeCubes([make_cube(x=200.0)])
        behavior = self.make(pycozmo.cube_behaviors.BehaviorCheckForStackAtInterval)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertEqual(self.cli.played, [])


class TestBehaviorKnockOverCubes(BehaviorTestCase):
    """ Grabs a block, rams the stack, and tells success from failure by movement. """

    CONF = {"minimumStackHeight": 3,
            "reachForBlockTrigger": "KnockOverGrabAttempt",
            "knockOverEyesTrigger": "KnockOverEyes",
            "knockOverSuccessTrigger": "KnockOverSuccess",
            "knockOverFailureTrigger": "KnockOverFailure",
            "knockOverPutDownTrigger": "PutDownBlockPutDown"}

    def setup_stack(self):
        self.cli.cubes = FakeCubes([make_cube(x=100.0),
                                    make_cube(x=300.0, z=22.5),
                                    make_cube(x=300.0, z=67.5),
                                    make_cube(x=300.0, z=112.5)])

    def test_knocked_stack_celebrates(self):
        self.setup_stack()
        top = self.cli.cubes[3]
        behavior = self.make(pycozmo.cube_behaviors.BehaviorKnockOverCubes, self.CONF)
        self.assertTrue(behavior.wants_to_run())
        real_drive = self.cli.drive_straight

        def ram(distance, speed=100.0, wait=True, timeout=None):
            top.pose = mock.Mock(x=360.0, y=0.0, z=22.5)
            return real_drive(distance, speed, wait=wait, timeout=timeout)

        with mock.patch("pycozmo.cube_handling.pick_up_cube", return_value=True), \
                mock.patch.object(self.cli, "drive_straight", side_effect=ram):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=20.0))
        self.assertEqual(self.cli.played,
                         ["KnockOverGrabAttempt", "KnockOverEyes", "KnockOverSuccess",
                          "PutDownBlockPutDown"])

    def test_standing_stack_commiserates(self):
        self.setup_stack()
        behavior = self.make(pycozmo.cube_behaviors.BehaviorKnockOverCubes, self.CONF)
        with mock.patch("pycozmo.cube_handling.pick_up_cube", return_value=True):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=20.0))
        self.assertEqual(self.cli.played,
                         ["KnockOverGrabAttempt", "KnockOverEyes", "KnockOverFailure",
                          "PutDownBlockPutDown"])

    def test_no_stack_nothing_to_knock(self):
        self.cli.cubes = FakeCubes([make_cube(x=100.0)])
        behavior = self.make(pycozmo.cube_behaviors.BehaviorKnockOverCubes)
        self.assertFalse(behavior.wants_to_run())


class TestBehaviorRespondPossiblyRoll(BehaviorTestCase):
    """ An upright block gets the happy answer, a rolled one the frustrated one. """

    def test_upright_block_gets_the_happy_answer(self):
        self.cli.cubes = FakeCubes([make_cube(x=200.0)])
        behavior = self.make(pycozmo.cube_behaviors.BehaviorRespondPossiblyRoll)
        self.assertTrue(behavior.wants_to_run())
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli))
        self.assertEqual(self.cli.played, ["BuildPyramidFirstBlockUpright"])

    def test_rolled_block_gets_the_frustrated_answer(self):
        self.cli.cubes = FakeCubes([make_cube(
            x=200.0, up_axis=pycozmo.protocol_encoder.UpAxis.XPositive)])
        behavior = self.make(pycozmo.cube_behaviors.BehaviorRespondPossiblyRoll)
        behavior.activate()
        self.assertTrue(complete_until_done(self, self.cli))
        self.assertEqual(self.cli.played, ["BuildPyramidFirstBlockOnSide"])

    def test_nothing_known_nothing_to_say(self):
        behavior = self.make(pycozmo.cube_behaviors.BehaviorRespondPossiblyRoll)
        self.assertFalse(behavior.wants_to_run())


def make_laser_image(x=100, y=50, size=5, width=320, height=240):
    """ A dark camera frame with a saturated square where the laser dot sits. """
    image = Image.new("L", (width, height))
    image.paste(255, (x, y, x + size, y + size))
    return image


class TestFindLaserSpot(BehaviorTestCase):
    """ A small saturated patch is a laser dot; darkness and floodlights are not. """

    def test_bright_dot_is_found(self):
        spot = pycozmo.explore_behaviors.find_laser_spot(make_laser_image())
        self.assertEqual(spot, (102.0, 52.0, 25))

    def test_dark_frame_finds_nothing(self):
        self.assertIsNone(pycozmo.explore_behaviors.find_laser_spot(
            Image.new("L", (320, 240))))

    def test_flooded_frame_finds_nothing(self):
        image = Image.new("L", (320, 240), 255)
        self.assertIsNone(pycozmo.explore_behaviors.find_laser_spot(image))


class TestBehaviorTrackLaser(BehaviorTestCase):
    """ Nothing seen ends the hunt; a centered dot earns a lunge, an off-center one a turn. """

    CONF = {"params": {"maxFailures": 1}}

    def test_nothing_seen_ends_the_hunt(self):
        behavior = self.make(pycozmo.explore_behaviors.BehaviorTrackLaser, self.CONF)
        with mock.patch.object(behavior, "_snap", return_value=None):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=10.0))
        self.assertEqual(len(self.cli.turns), 12)
        self.assertEqual(self.cli.played, ["LaserGetOut"])

    def test_centered_spot_gets_pounced(self):
        from pycozmo.camera import DEFAULT_CALIBRATION
        spot = (DEFAULT_CALIBRATION.cx, 120.0, 10)
        behavior = self.make(pycozmo.explore_behaviors.BehaviorTrackLaser, self.CONF)
        with mock.patch.object(behavior, "_snap", side_effect=[spot] + [None] * 12):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=10.0))
        self.assertEqual(self.cli.played,
                         ["LaserAcknowledge", "LaserPounce", "LaserDriveStart",
                          "LaserDriveEnd", "LaserGetOut"])
        self.assertEqual(self.cli.drives, [150.0])

    def test_off_center_spot_turns_to_face_it(self):
        behavior = self.make(pycozmo.explore_behaviors.BehaviorTrackLaser, self.CONF)
        with mock.patch.object(behavior, "_snap", side_effect=[(0.0, 0.0, 10)] + [None] * 12):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=10.0))
        self.assertEqual(len(self.cli.turns), 13)
        self.assertEqual(self.cli.drives, [])
        self.assertEqual(self.cli.played, ["LaserAcknowledge", "LaserGetOut"])


class TestBehaviorFeedingSearchForCube(BehaviorTestCase):
    """ A cube in sight ends the search at once; an empty floor gets a full turn. """

    def test_cube_in_sight_finishes_at_once(self):
        behavior = self.make(pycozmo.feeding_behaviors.BehaviorFeedingSearchForCube)
        with mock.patch("pycozmo.feeding_behaviors.usable_cubes", return_value=[mock.Mock()]):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertEqual(self.cli.turns, [])
        self.assertEqual(self.cli.played,
                         ["FeedingSearchRequest", "FeedingReactToSeeCube_Normal"])

    def test_empty_floor_searches_all_round(self):
        behavior = self.make(pycozmo.feeding_behaviors.BehaviorFeedingSearchForCube)
        with mock.patch("pycozmo.feeding_behaviors.usable_cubes", return_value=[]):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=15.0))
        self.assertEqual(len(self.cli.turns), 12)
        self.assertEqual(self.cli.played, ["FeedingSearchRequest", "FeedingSearchFailure"])

    def test_severe_energy_asks_severely(self):
        behavior = self.make(pycozmo.feeding_behaviors.BehaviorFeedingSearchForCube)
        behavior.needs = StubNeeds({"Energy": "Critical"})
        with mock.patch("pycozmo.feeding_behaviors.usable_cubes", return_value=[mock.Mock()]):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=3.0))
        self.assertEqual(self.cli.played,
                         ["FeedingSearchRequest_Severe", "FeedingReactToSeeCube_Severe"])


class TestBehaviorFeedingEat(BehaviorTestCase):
    """ A shaken cube counts as fed; a still one does not; no cube leaves quietly. """

    CONF = {"watchForShake_sec": 0.4}

    def run_eat(self, testcase, cubes, timeout=10.0):
        behavior = testcase.make(pycozmo.feeding_behaviors.BehaviorFeedingEat, self.CONF)
        with mock.patch("pycozmo.feeding_behaviors.usable_cubes", return_value=cubes), \
                mock.patch("pycozmo.cube_handling.go_to_cube", return_value=True):
            behavior.activate()
            testcase.assertTrue(complete_until_done(testcase, testcase.cli, timeout=timeout))
        return behavior

    def test_no_cube_leaves_quietly(self):
        self.run_eat(self, [])
        self.assertEqual(self.cli.played, [])
        self.assertEqual(self.cli.drives, [])

    def test_still_cube_is_not_enough(self):
        cube = make_cube()
        self.run_eat(self, [cube])
        self.assertEqual(self.cli.played,
                         ["FeedingPlaceLiftOnCube_Normal", "FeedingAteNotFullEnough_Normal"])
        self.assertEqual(self.cli.drives, [20.0])

    def test_shaken_cube_counts_as_fed(self):
        cube = make_cube()
        behavior = self.make(pycozmo.feeding_behaviors.BehaviorFeedingEat, self.CONF)
        before = mock.Mock(x=0.0, y=0.0, z=0.0)
        after = mock.Mock(x=50.0, y=0.0, z=0.0)
        with mock.patch("pycozmo.feeding_behaviors.usable_cubes", return_value=[cube]), \
                mock.patch("pycozmo.cube_handling.go_to_cube", return_value=True), \
                mock.patch.object(behavior, "_pose_of", side_effect=[before, after]):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=10.0))
        self.assertEqual(self.cli.played,
                         ["FeedingPlaceLiftOnCube_Normal", "FeedingReactToShake_Normal",
                          "FeedingAteFullEnough_Normal"])
        self.assertEqual(self.cli.drives, [20.0, 20.0])

    def test_severe_energy_uses_severe_triggers(self):
        cube = make_cube()
        behavior = self.make(pycozmo.feeding_behaviors.BehaviorFeedingEat, self.CONF)
        behavior.needs = StubNeeds({"Energy": "Critical"})
        with mock.patch("pycozmo.feeding_behaviors.usable_cubes", return_value=[cube]), \
                mock.patch("pycozmo.cube_handling.go_to_cube", return_value=True):
            behavior.activate()
            self.assertTrue(complete_until_done(self, self.cli, timeout=10.0))
        self.assertEqual(self.cli.played,
                         ["FeedingPlaceLiftOnCube_Severe", "FeedingAteNotFullEnough_Severe"])


class TestEmotionEvents(BehaviorTestCase):
    """ Behaviors that shift the mood as well as playing an animation. """

    def test_frustration_posts_the_configured_emotion_event(self):
        behavior = self.make(pycozmo.behavior.BehaviorReactToFrustration,
                             {"anim": "FrustratedByFailure",
                              "finalEmotionEvent": "FinishedMinorFrustration"},
                             behavior_id="ReactToFrustrationMinor")
        behavior.activate()
        self.assertEqual(self.posted_emotion_events(), [])
        self.complete_animation()
        # Only once it is over, which is what "final" means in the configuration.
        self.assertEqual(self.posted_emotion_events(), ["FinishedMinorFrustration"])
        self.assertDone()

    def test_frustration_without_an_emotion_event(self):
        behavior = self.make(pycozmo.behavior.BehaviorReactToFrustration,
                             {"anim": "FrustratedByFailure"})
        behavior.activate()
        self.complete_animation()
        self.assertEqual(self.posted_emotion_events(), [])
        self.assertDone()

    def test_unexpected_movement_costs_confidence(self):
        behavior = self.make(pycozmo.behavior.BehaviorReactToUnexpectedMovement)
        behavior.activate()
        self.complete_animation()
        self.assertEqual(self.posted_emotion_events(), ["ReactToUnexpectedMovement"])
        self.assertDone()

    def test_driving_off_the_charger_builds_confidence(self):
        behavior = self.make(pycozmo.behavior.BehaviorDriveOffCharger)
        behavior.activate()
        self.addCleanup(behavior.deactivate)
        self.assertEqual(self.posted_emotion_events(), ["DriveOffCharger"])

    def test_a_plain_reaction_posts_none(self):
        behavior = self.make(pycozmo.behavior.BehaviorReactToPickup)
        behavior.activate()
        self.complete_animation()
        self.assertEqual(self.posted_emotion_events(), [])
