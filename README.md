# Rainbow Six Siege Weapon Statistics
Provides detailed statistics for all weapons in Tom Clancy's Rainbow Six Siege.

After every game patch a spreadsheet of the important stats is [released](https://github.com/hanslhansl/rainbow-six-siege-weapon-statistics/releases). The most recent version can be found on [Google Sheets](https://docs.google.com/spreadsheets/d/1QgbGALNZGLlvf6YyPLtywZnvgIHkstCwGl1tvCt875Q)/[Google Drive](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1KitQsZksdVP9YPDInxK3xE2gtu1mpUxV5_PNyE8sSm-vFINdbiL8vo9RA2CRSIbIUePLVA1GCTWZ/pubhtml).

# About how the data is collected
Generally, I am trying to collect as much of the data myself and by hand because Ubisoft has proven to be an unreliable source for statistics. At the same time I am trying to cut corners where possible.

## Time measuring
I am measuring time intervals by recording the game and going off of frame time. I am recording at 120 fps which means that, in theory, every time measurement can deviate up to $Δt={1\\over120} s\\approx0.0083 s$ from the actual value. In conclusion, this means that every time related stat (e.g. fire rate, dps, ads time, ...) should be taken with a grain of salt.

## Damage per bullet
I am measuring the damage values in the in-game Shooting Range which limits my measurements to distances of 5 to 40 meters (thanks Ubi). This would mean that I need to measure 36 distinct damage values (from 5 to 40 meters) per weapon. However:
* Damage in R6S never increases over distance. It only decreases or stagnates. Therefor, if the damage at distance A is equal to the damage at distance B the damage at all distances inbetween A and B is also equal. Because of this I don't have to measure at all distances.
* Most weapons in R6S have exactly one damage drop-off interval. Up until drop-off start they deal the base damage. From drop-off start to drop-off end they progressively lose damage over distance. From drop-off end on they deal the final damage. Because of this the damage up until 5 meters (which can't be measured in the shooting range) can be assumed to equal the damage at 5 meters.

The unmeasured but deducible damage values are supplemented automatically by the Python script.

I am using the distance value displayed on the shooting range panel to measure the distance to the dummy. This value (just like yellow ping btw.) is a rounded up integer value (e.g 7.1 m is displayed as 8 m). To position yourself exactly $n$ meters away from the dummy you have to stand where the displayed value is about to switch from $n$ to $n+1$. To measure the weapon damages at the exact distances and not inbetween I am trying to be as precise with this as possible. 

### Shotguns
Shotguns are the exception, they have two damage drop-off intervals, usually $\[5; 6]$ and $\[10; 12]$. Because the first interval starts at 5 meters it is not possible to verify the damage below 5 meters in the Shooting Range. Therefor I have to trust Ubisoft on the ingame shotgun damage stats for distances less than 5 meters.

### Hit areas
There are three body hit areas in R6S: the head, the torso and the limbs. The torso has a damage multiplier of $1$ and therefor receives the base damage. This is the value I am measuring. Most weapons deal infinite damage to the head. The only exception are shotguns which deal $150\\%$ of the base damage to the head. Limb damage is a bit more complicated. As of now I believe most weapons deal $75\\%$ of the base damage to limbs. However, there exists at least one exception to this rule, Kali's CSRX 300, which deals $60\\%$ to limbs. At some point I will test all weapon's limb damage multiplier. soon (tm).

## Bullets per shot - Pellet count
This value is displayed in the shooting range. Most weapons shoot exactly one bullet per shot. The only exception are shotguns, most of which shoot 8 bullets per shot. For shotguns this metric is also called pellet count.

## Damage per shot
The damage per shot is calculated as $DamagePerShot = Damage \* Pellets$. No measuring necessary. Only for shotguns does this value differ from the damage per bullet.

## Fire rate
Currently, for all fully automatic weapons I am using the fire rates listed in-game. For all other weapons I measure the fire rates myself. I am doing this by emptying a magazine of size $n$ and measuring the time $t$ (in seconds) between the in-game ammo counter decreasing the first time and the ammo counter reaching 0. The fire rate in rounds per seconds is calculated as ${n-1 \\over t} rps$ and in rounds per minute as $60{n-1 \\over t} rpm$.

As [already mentioned](#time-measuring) my time measurements can vary up to $Δt$ from the real values. The actual fire rates are therefor in the interval defined by $60{n-1 \\over t \\pm Δt} rpm$. For example a measured fire rate of 800 rpm for a weapon with 31 bullets per magazine would mean an actual fire rate of something inbetween 797 rpm and 803 rpm. Because of this innaccuracy I am usually rounding the fire rate to a reasonable integer within said interval (e.g. measured 433.47 rpm become 430 rpm).

## Damage per second - DPS
The damage per second is calculated as $DPS = DmgPerShot \* RPS = DmgPerShot \* RPM / 60$. No measuring necessary.

## Bullets to down or kill - BTDOK
For a target with $x$ hp the BTDOK is calculated as $\\lceil {x \\over Damage} \\rceil$. No measuring necessary.

## Shots to down or kill - STDOK
For a target with $x$ hp the STDOK is calculated as $\\lceil {x \\over Damage per shot} \\rceil$. No measuring necessary.

## Time to down or kill - TTDOK
For a target with $x$ hp the TTDOK in milliseconds is calculated as ${STDOK - 1 \\over rpms}$. No measuring necessary.

## Magazine capacity
Most weapons have, in addition to the bullets loaded in the magazine, one bullet loaded in the chamber. For those weapons this value is displayed as $Capacity+1$ (e.g. $30+1$, $20+1$, etc.). For all other weapons without a bullet loaded in the chamber this value is displayed as $Capacity+0$ (e.g. $100+0$, $80+0$).

## Aim down sight time - ADS time
The time in seconds it takes to aim down sight while standing still. The laser attachment increases the ads speed by $10\\%$. The reduced ads time with laser is calculated as $ads \\over 1.1$.

## Reload time
soon (tm)
