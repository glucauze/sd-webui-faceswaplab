from typing import Tuple
from numpy import uint8
from numpy.typing import NDArray
from insightface.app.common import Face as IFace
from PIL import Image

PILImage = Image.Image
CV2ImgU8 = NDArray[uint8]
Face = IFace
BoxCoords = Tuple[int, int, int, int]
