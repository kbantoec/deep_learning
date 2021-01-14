from cycler import cycler
from itertools import cycle

colors = {'green': '#59C899', 
          'blue': '#6370F1', 
          'orange': '#F3A467', 
          'purple': '#A16AF2', 
          'light_blue': '#5BCCC7', 
          'red': '#DF6046'}

cmap = cycle(colors.values())

mpl_cycler = cycler(color=colors.values())