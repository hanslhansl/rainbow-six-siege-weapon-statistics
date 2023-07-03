#  R6S Weapon Statistics
Provides detailed information for all weapons in R6S. This includes the weapon base damage at distances up to 40 meters, fire rate, damage per second and shots to down or kill.


# About how the data is collected
Generally speaking I am trying to collect as much of the data myself and by hand because Ubisoft has proven to be an unreliable source for statistics. At the same time I am trying cut corners where possible. This project 
## Damage stats
## Fire rate
-incomplete- Currently, for all fully automatic weapons the fire rates are the ones listed in-game. For all other weapons I meassure the fire rates myself. I am doing this by emptying a magazine of size $n$ and meassuring the time $t$ (in milliseconds) between the ammo counter decreasing the first time and the ammo counter reaching 0. The fire rate in rounds per milliseconds calculates as ${n-1 \over t}rpms$, in rounds per second as $1000{n-1 \over t}rps$ and in rounds per minute as $60000{n-1 \over t}rpm$. It is important to subract $1$ from $n$ because the first bullet in a spray leaves the weapon immediatelly after pulling the trigger and therefor doesn't contribute to the meassured time. Without subtracting 1 you would get different values for the fire rate depending on how many bullets you shot which doesn't make sense.

My meassurements are taken at 60 fps meaning they can deviate up to ${1\over60}s\approx0.0167s=16.7ms$ from the actual values. The actual fire rates are therefor in the interval defined by $60000{n-1 \over t \pm 16.7}rpm$. For example a meassured fire rate of 800 rpm for a weapon with 31 bullets per magazine would mean an actual fire rate of something inbetween 794 to 806 rpm.

At some point I will meassure all fully automatic weapon's fire rates as well. Just to be sure. soon (tm).

## Reload time
soon (tm)
## ADS time
soon (tm)
## Damage per second
The damage per seconds calculates as $DPS = Damage * RPS = Damage * RPM * 60$. No meassuring necessary.
