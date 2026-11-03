# TODO — Cozmo pending items

State: FW 2381, all 76 Anki behavior classes restored, full suite green
except the pre-existing `json_loader` filesystem failure. Robot docked and
charging (~4.4 V). Evening rule: set volume to 5000/65535 right after every
connect (the robot resets it on connect).

## Needs the user (hardware)

- [ ] Cubes are dead: none light up, none connect. Replace the N-type 1.5 V
  alkaline batteries (LR1 / 910A / E90, + side up, compartment on the bottom
  under a screw). Fresh battery flashes the LEDs on install. Then set lit
  cubes near the awake Cozmo; they join on their own.
- [ ] If a fresh battery still gives no light, that cube has failed — source
  secondhand cubes.

## Once cubes are seen (re-probe first: all three should report connected)

- [ ] `FeedingSearchForCube` + `FeedingEat` on-robot with a cube.
- [ ] `GuardDog`, `Bouncer`, `OnboardingShowCube` on-robot with cubes.
- [ ] Pyramid/stack behaviors (`BuildPyramid`, base, tall-stack, knock-over)
  on-robot with cubes.
- [ ] `TrackLaser` needs a laser pointer (none available yet).

## Already verified on-robot (volume 5000, no cubes)

- [x] `FactoryCentroidExtractor`, `LiftLoadTest`, `ThinkAboutBeacons`,
  `DevTurnInPlaceTest` — all pass; notes in `docs/wsl_firmware_update.md`.
- [x] Song playback, Brain idle/docking behavior, `go_to_charger` — see the
  same doc. Docking needs the charger marker lit from ~45° to the side.
