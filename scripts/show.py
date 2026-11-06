import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

image, model = sys.argv[1], sys.argv[2]
hdu = fits.open(image)[0]
wcs = WCS(hdu.header).celestial
ra, dec = np.loadtxt(model, usecols=(1, 2), unpack=True)

ax = plt.subplot(projection=wcs)
ax.imshow(hdu.data[0, 0], origin="lower", cmap="inferno")
ax.scatter(ra, dec, s=150, fc="none", ec="cyan", transform=ax.get_transform("world"))
plt.savefig(image.replace(".fits", ".png"))
print("Successfully generated PNG for:", image)
