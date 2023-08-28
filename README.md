#  R6S Weapon Statistics
Provides detailed statistics for all weapons in Tom Clancy's Rainbow Six Siege. This includes the weapon's damage at distances up to 40 meters, fire rate, damage per second, shots to down or kill, reload time and ADS time. All data was collected during operation Dread Factor. A spreadsheet of the important stats can be found [here](https://docs.google.com/spreadsheets/d/1QgbGALNZGLlvf6YyPLtywZnvgIHkstCwGl1tvCt875Q/edit?usp=sharing).

Currently a few damage stats are still missing. I am waiting for the release of operation Heavy Mettle to collect the missing stats as shotgun damage will change drastically with that patch.
# About how the data is collected
Generally speaking I am trying to collect as much of the data myself and by hand because Ubisoft has proven to be an unreliable source for statistics. At the same time I am trying cut corners where possible.
## Time measuring
I am measuring time intervals by recording the game, opening the recording in a video editing program and measuring the time stamps. This method is by far not as accurate as would want it to be but I don't know of any alternative. I am recording my game at 60 fps which means that, in theory, every time measurement can deviate up to ${1\over60}s\approx0.0167s=16.7ms$ from the actual value. Unfortunatelly I have come to the conclusion that in some cases my measurements deviate consideribly more than just one sixtieth of a second. I am not sure why that is and how to fix it. Higher frame rates could solve this but my PC limits me to recording at 60 fps. In conclusion this means that every time related stat (e.g. fire rate, DPS, ads time, ...) should be taken with a grain of salt.
## Damage stats
I am measuring the damage values in the in-game Shooting Range which limits my measurements to distances of 5 to 40 meters (thanks Ubi). I am not using the Yellow Ping feature to measure the distance to the dummy but rather the distance value displayed on the shooting range panel. This value (just like yellow ping btw.) is a rounded up integer value (e.g 7.1m is displayed as 8m). To position yourself e.g. 8 meters away from the dummy you have to stand where the displayed distance switches from 8 to 9 meters. This is exactly what I am trying to achieve: Measure the weapon damages at the exact distances and not inbetween. This means I am measuring one damage value per integer distance for every weapon. 

Experience has shown that damage does not increase over distance. It only decreases or stagnates. Therefor, if the damage at distance A is equal to the damage at distance B the damage at all distances inbetween A and B is also equal to said damage. I am using this knowledge so that I don't have to measure at all distances. The unmeasured but known damages are supplemented automatically by the Python script.

Experience has also shown that (almost) all weapons have exactly one damage drop-off interval. Up until drop-off start (dos) they deal the base damage. From dos to drop-off end (doe) they progressively lose damage over distance and from doe on they do the final damage. Because of this the damage up until 5 meters (which can't be measured in the shooting range) can be assumed to equal the damage at 5 meters. To prevent errors in case the damage drop-off interval reaches into the first 5 meters I am only applying this assumption if the measured damages at 5, 6 and 7 meters are equal. If that isn't the case I leave the damages up until 5 meters blank and test them in a custom game. It should be said that even with this precautionary measure it is still possible for me to overlook a damage drop within the first 5 meters so those values should be taken with a grain of salt.

I am measuring the weapon's damage to the torso as this body region has a damage multiplier of 1. Most weapons deal infinite damage to the head. The only exception are shotguns which deal the same damage to head as they deal to the torso. As of now I believe most weapons to deal 75% of the torso damage to limbs. There exists at least one exception to this rule though, Kali's CSRX 300, which deals 60% to limbs. At some point I will test all weapon's limb damage multiplier. soon (tm).
## Fire rate
Currently, for all fully automatic weapons the fire rates are the ones listed in-game. For all other weapons I measure the fire rates myself. I am doing this by emptying a magazine of size $n$ and measuring the time $t$ (in milliseconds) between the ammo counter decreasing the first time and the ammo counter reaching 0. The fire rate in rounds per milliseconds calculates as ${n-1 \over t}rpms$, in rounds per second as $1000{n-1 \over t}rps$ and in rounds per minute as $60000{n-1 \over t}rpm$. It is important to subract 1 from n because the first bullet in a spray leaves the weapon immediatelly after pulling the trigger and therefor doesn't contribute to the measured time. Without subtracting 1 you would get different values for the fire rate depending on how many bullets you shot which doesn't make sense.

As mentioned [previously](#time-measuring) my time measurements can vary up to 16.7ms from the real values. The actual fire rates are therefor in the interval defined by $60000{n-1 \over t \pm 16.7}rpm$. For example a measured fire rate of 800 rpm for a weapon with 31 bullets per magazine would mean an actual fire rate of something inbetween 794 rpm and 806 rpm. Because of this innaccuracy I am usually rounding the fire rate to a reasonable integer within said interval (e.g. measured 433.47 rpm become 430 rpm).

At some point I will measure all fully automatic weapon's fire rates as well. Just to be sure. soon (tm).
## Reload time
soon (tm)
## Aim down sight time - ADS time
soon (tm)
## Damage per second - DPS
The damage per second calculates as $DPS = Damage * RPS = Damage * RPM / 60$. No measuring necessary.
## Shots to down or kill - STDOK
For a target with $x$ hp the STDOK calculate as $\lceil {x \over Damage} \rceil$. No measuring necessary.
## Time to down or kill - TTDOK
For a target with $x$ hp the TTDOK in milliseconds calculate as ${STDOK \over rpms}$. No measuring necessary.
## Bullets per shot - Pellet count
The shooting range displays the bullets per shot. Most weapons shoot exactly one bullet per shot. The only exception are shotguns, most of which shoot 8 bullets per shot. For weapons that shoot multiple bullets per shot this metric is also called pellet count.
## Magazine capacity
Most weapons have one bullet loaded in the chamber. For those this value is displayed as $capacity+1$ (e.g. $30+1$, $20+1$, etc.). For all other weapons it is displayed as $capacity+0$ (e.g. $100+0$, $80+0$). soon (tm)
