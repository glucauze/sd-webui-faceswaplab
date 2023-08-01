from scripts.faceswaplab_swapping import swapper
import base64
import io
from dataclasses import dataclass
from typing import List, Optional, Set, Union
import gradio as gr
from insightface.app.common import Face
from PIL import Image
from scripts.faceswaplab_utils.imgutils import pil_to_cv2
from scripts.faceswaplab_utils.faceswaplab_logging import logger
from scripts.faceswaplab_utils import face_checkpoints_utils
from scripts.faceswaplab_inpainting.faceswaplab_inpainting import InpaintingOptions
from client_api import api_utils


@dataclass
class FaceSwapUnitSettings:
    # ORDER of parameters is IMPORTANT. It should match the result of faceswap_unit_ui

    # The image given in reference
    source_img: Optional[Union[Image.Image, str]]
    # The checkpoint file
    source_face: Optional[str]
    # The batch source images
    _batch_files: Optional[Union[gr.components.File, List[Image.Image]]]
    # Will blend faces if True
    blend_faces: bool
    # Enable this unit
    enable: bool
    # Use same gender filtering
    same_gender: bool
    # Sort faces by their size (from larger to smaller)
    sort_by_size: bool
    # If True, discard images with low similarity
    check_similarity: bool
    # if True will compute similarity and add it to the image info
    _compute_similarity: bool

    # Minimum similarity against the used face (reference, batch or checkpoint)
    min_sim: float
    # Minimum similarity against the reference (reference or checkpoint if checkpoint is given)
    min_ref_sim: float
    # The face index to use for swapping
    _faces_index: str
    # The face index to get image from source
    reference_face_index: int

    # Swap in the source image in img2img (before processing)
    swap_in_source: bool
    # Swap in the generated image in img2img (always on for txt2img)
    swap_in_generated: bool
    # Pre inpainting configuration (Don't use optional for this or gradio parsing will fail) :
    pre_inpainting: InpaintingOptions
    # Post inpainting configuration (Don't use optional for this or gradio parsing will fail) :
    post_inpainting: InpaintingOptions

    @staticmethod
    def from_api_dto(dto: api_utils.FaceSwapUnit) -> "FaceSwapUnitSettings":
        """
        Converts a InpaintingOptions object from an API DTO (Data Transfer Object).

        :param options: An object of api_utils.InpaintingOptions representing the
                        post-processing options as received from the API.
        :return: A InpaintingOptions instance containing the translated values
                from the API DTO.
        """
        return FaceSwapUnitSettings(
            source_img=api_utils.base64_to_pil(dto.source_img),
            source_face=dto.source_face,
            _batch_files=dto.get_batch_images(),
            blend_faces=dto.blend_faces,
            enable=True,
            same_gender=dto.same_gender,
            sort_by_size=dto.sort_by_size,
            check_similarity=dto.check_similarity,
            _compute_similarity=dto.compute_similarity,
            min_ref_sim=dto.min_ref_sim,
            min_sim=dto.min_sim,
            _faces_index=",".join([str(i) for i in (dto.faces_index)]),
            reference_face_index=dto.reference_face_index,
            swap_in_generated=True,
            swap_in_source=False,
            pre_inpainting=InpaintingOptions.from_api_dto(dto.pre_inpainting),
            post_inpainting=InpaintingOptions.from_api_dto(dto.post_inpainting),
        )

    @property
    def faces_index(self) -> Set[int]:
        """
        Convert _faces_index from str to int
        """
        faces_index = {
            int(x) for x in self._faces_index.strip(",").split(",") if x.isnumeric()
        }
        if len(faces_index) == 0:
            return {0}

        logger.debug("FACES INDEX : %s", faces_index)

        return faces_index

    @property
    def compute_similarity(self) -> bool:
        return self._compute_similarity or self.check_similarity

    @property
    def batch_files(self) -> List[gr.File]:
        """
        Return empty array instead of None for batch files
        """
        return self._batch_files or []

    @property
    def reference_face(self) -> Optional[Face]:
        """
        Extract reference face (only once and store it for the rest of processing).
        Reference face is the checkpoint or the source image or the first image in the batch in that order.
        """
        if not hasattr(self, "_reference_face"):
            if self.source_face and self.source_face != "None":
                try:
                    logger.info(f"loading face {self.source_face}")
                    face = face_checkpoints_utils.load_face(self.source_face)
                    self._reference_face = face
                except Exception as e:
                    logger.error("Failed to load checkpoint  : %s", e)
                    raise e
            elif self.source_img is not None:
                if isinstance(self.source_img, str):  # source_img is a base64 string
                    if (
                        "base64," in self.source_img
                    ):  # check if the base64 string has a data URL scheme
                        base64_data = self.source_img.split("base64,")[-1]
                        img_bytes = base64.b64decode(base64_data)
                    else:
                        # if no data URL scheme, just decode
                        img_bytes = base64.b64decode(self.source_img)
                    self.source_img = Image.open(io.BytesIO(img_bytes))
                source_img = pil_to_cv2(self.source_img)
                self._reference_face = swapper.get_or_default(
                    swapper.get_faces(source_img), self.reference_face_index, None
                )
                if self._reference_face is None:
                    logger.error("Face not found in reference image")
            else:
                self._reference_face = None

        if self._reference_face is None:
            logger.error("You need at least one reference face")
            raise Exception("No reference face found")

        return self._reference_face

    @property
    def faces(self) -> List[Face]:
        """_summary_
        Extract all faces (including reference face) to provide an array of faces
        Only processed once.
        """
        if self.batch_files is not None and not hasattr(self, "_faces"):
            self._faces = (
                [self.reference_face] if self.reference_face is not None else []
            )
            for file in self.batch_files:
                if isinstance(file, Image.Image):
                    img = file
                else:
                    img = Image.open(file.name)

                face = swapper.get_or_default(
                    swapper.get_faces(pil_to_cv2(img)), 0, None
                )
                if face is not None:
                    self._faces.append(face)
        return self._faces

    @property
    def blended_faces(self) -> Face:
        """
        Blend the faces using the mean of all embeddings
        """
        if not hasattr(self, "_blended_faces"):
            self._blended_faces = swapper.blend_faces(self.faces)

        return self._blended_faces
