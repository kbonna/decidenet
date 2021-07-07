import matplotlib.colors as mc
import matplotlib as mpl
import colorsys

from cycler import cycler

BLUE_BLACK = "#182C4B"
BLUE_DARK = "#315895"
BLUE = "#5E88CA"
BLUE_LIGHT = "#93AFDB"
BLUE_PALE = "#C8D6EC"

ORANGE_BLACK = "#563404"
ORANGE_DARK = "#A96707"
ORANGE = "#F49915"
ORANGE_LIGHT = "#F7BA62"
ORANGE_PALE = "#FBDCAF"

# task stimuli colors
BLUE_TASK = "#56B4E9" 
YELLOW_TASK = "#F0E442" 

MPL_STYLES = {
    "font.size": 19, 
    "font.family": "Helvetica",
    "lines.linewidth": 2, 
    
    "axes.prop_cycle": cycler(color=[BLUE, ORANGE, "#7fb685", "#f97068", "#ce84ad"]),
    "axes.titlesize": 16,
    "axes.linewidth": 2,
    "axes.spines.top": True,
    "axes.spines.right": True,

    "xtick.direction": "out", 
    "xtick.major.size": 7,
    "xtick.major.width": 2,
    "xtick.minor.size": 4,
    "xtick.minor.width": 2,
    "xtick.top": False,
    "ytick.direction": "out", 
    "ytick.major.size": 7,
    "ytick.major.width": 2,
    "ytick.minor.size": 4,
    "ytick.minor.width": 2,
    "ytick.right": False,
    
    "legend.frameon": False,
    "legend.fontsize": 16,
    "legend.framealpha": 0,
}

def lighten_color(color, amount=0.5):
    """ Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
        >> lighten_color("g", 0.3)
        >> lighten_color("#F034A3", 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return mc.to_hex(c)

POWER_COLORS = {
    "Blue": "#0000ff",
    "Teal": "#008080",
    "White": "#ffffff",
    "Brown": "#a52a2a",
    "Cyan": "#00ffff", 
    "Orange": "#ffa500",
    "Black": "#000000",
    "Gray": "#808080",
    "Yellow": "#ffff00",
    "Green": "#00ff00",
    "Red": "#ff0000",
    "Purple": "#800080",
    "Pale blue": "#87ceeb",
    "Pink": "#ff1493",
}



