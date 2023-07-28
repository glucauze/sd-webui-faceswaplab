---
layout: page
title: Features
permalink: /features/
---

+ **Face Unit Concept**: Similar to controlNet, the program introduces the concept of a face unit. You can configure up to 10 units (3 units are the default setting) in the program settings (sd).

![](/assets/images/face_units.png)

+ **Vladmantic and a1111 Support**

+ **Batch Processing**

+ **Inpainting**: supports "only masked" and mask inpainting.

+ **Performance Improvements**: The overall performance of the software has been enhanced.

+ **FaceSwapLab Tab**  providing various tools.

![](/assets/images/tab.png)

+ **FaceSwapLab Settings**: FaceSwapLab settings are now part of the sd settings. To access them, navigate to the sd settings section.

![](/assets/images/settings.png)

+ **Face Reuse Via Checkpoints**: The FaceTools tab now allows creating checkpoints, which facilitate face reuse. When a checkpoint is used, it takes precedence over the reference image, and the reference source image is discarded.

![](/assets/images/checkpoints.png)
![](/assets/images/checkpoints_use.png)

+ **Gender Detection**: The program can now detect gender based on faces.

![](/assets/images/gender.png)

+ **Face Combination (Blending)**: Multiple versions of a face can be combined to enhance the swapping result. This blending happens during checkpoint creation.

![](/assets/images/blend_face.png)
![](/assets/images/testein.png)

+ **Preserve Original Images**: You can opt to keep original images before the swapping process.

![](/assets/images/keep_orig.png)

+ **Multiple Face Versions for Replacement**: The program allows the use of multiple versions of the same face for replacement.

![](/assets/images/multiple_face_src.png)

+ **Face Similarity and Filtering**: You can compare faces against the reference and/or source images.

![](/assets/images/similarity.png)

+ **Face Comparison**: face comparison feature.

![](/assets/images/compare.png)

+ **Face Extraction**: face extraction with or without upscaling.

![](/assets/images/extract.png)

+ **Improved Post-Processing**: codeformer, gfpgan, upscaling.

![](/assets/images/post-processing.png)

+ **Post Inpainting**: This feature allows the application of image-to-image inpainting specifically to faces.

![](/assets/images/postinpainting.png)
![](/assets/images/postinpainting_result.png)

+ **Upscaled Inswapper**: The program now includes an upscaled inswapper option, which improves results by incorporating upsampling, sharpness adjustment, and color correction before face is merged to the original image.

![](/assets/images/upscalled_swapper.png)


+ **API with typing support** :

```python
import base64
import io
import requests
from PIL import Image
from client_utils import FaceSwapRequest, FaceSwapUnit, PostProcessingOptions, FaceSwapResponse, pil_to_base64

address = 'http:/127.0.0.1:7860'

# First face unit :
unit1 = FaceSwapUnit(
    source_img=pil_to_base64("../../references/man.png"), # The face you want to use
    faces_index=(0,) # Replace first face
)

# Second face unit :
unit2 = FaceSwapUnit(
    source_img=pil_to_base64("../../references/woman.png"), # The face you want to use
    same_gender=True,
    faces_index=(0,) # Replace first woman since same gender is on
)

# Post-processing config :
pp = PostProcessingOptions(
    face_restorer_name="CodeFormer",
    codeformer_weight=0.5,
    restorer_visibility= 1)

# Prepare the request
request = FaceSwapRequest (
    image = pil_to_base64("test_image.png"),
    units= [unit1, unit2],
    postprocessing=pp
)


result = requests.post(url=f'{address}/faceswaplab/swap_face', data=request.json(), headers={"Content-Type": "application/json; charset=utf-8"})
response = FaceSwapResponse.parse_obj(result.json())

for img, info in zip(response.pil_images, response.infos):
    img.show(title = info)


```