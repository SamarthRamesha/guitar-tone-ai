import numpy as np


def _clip(x, lo=0.0, hi=10.0):
    return round(float(min(max(x, lo), hi)), 2)


def perceptual_to_knobs(perceptual):
    """
    perceptual = {
        "saturation": 0..1,
        "brightness": 0..1,
        "mid_emphasis": 0..1,
        "low_end": 0..1
    }

    Returns REAL amp knob values (0–10 scale)
    """

    sat = perceptual["saturation"]
    bright = perceptual["brightness"]
    mids = perceptual["mid_emphasis"]
    low = perceptual["low_end"]

    #non linear gain
    #clean: stays low
    #metal: ramps FAST after ~0.5
    if sat < 0.4:
        gain = 1.5 + sat * 7.5
    else:
        gain = 3.5 + (sat ** 1.7) * 6.5

        #bass: low end punch without flub
    bass = 3.5 + low * 4.5
    if sat > 0.6:
        bass += 0.8   # metal push

    #mids clean: forward mids
    #high gain: scooped but NOT hollow
    if sat < 0.4:
        mid = 4.5 + mids * 3.5
    else:
        mid = 5.0 - mids * 2.2

    # treble: bright but not harsh
    treble = 3.5 + bright * 3.8
    if sat > 0.6:
        treble -= 0.8  # tame fizz

    #presence
    presence = 3.0 + bright * 4.0
    if sat > 0.6:
        presence += 0.7

    return {
        "gain": _clip(gain),
        "bass": _clip(bass),
        "mid": _clip(mid),
        "treble": _clip(treble),
        "presence": _clip(presence)
    }