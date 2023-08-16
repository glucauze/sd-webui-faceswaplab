from typing import Tuple
from numpy import uint8
from insightface.app.common import Face as IFace
from PIL import Image
import numpy as np

PILImage = Image.Image
CV2ImgU8 = np.ndarray[int, np.dtype[uint8]]
Face = IFace
BoxCoords = Tuple[int, int, int, int]
