import sys
import io
import fitz
import matplotlib.pyplot as plt
from PIL import Image

doc = fitz.open(sys.argv[1])
fig, axes = plt.subplots(1, len(doc), figsize=(12, 6))
if len(doc) == 1:
    axes = [axes]

for ax, page in zip(axes, doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    ax.imshow(Image.open(io.BytesIO(pix.tobytes("png"))))
    ax.axis("off")

plt.show()
