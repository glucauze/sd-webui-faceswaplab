# 1.2.5

Allow seed selection in inpainting.

# 1.2.4

Fix default settings by marking only managed field as do_not_save.

See the discussion here : https://github.com/glucauze/sd-webui-faceswaplab/issues/62

# 1.2.3

Speed up ui : change the way default settings are manage by not storing them in ui-config.json

Migration : YOU NEED TO recreate ui-config.json (delete) or at least remove any faceswaplab reference to be able to use default settings again.

See this for explainations : https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/6109

# 1.2.2

+ Add NSFW filter option in settings (1 == disable)
+ Improve install speed
+ Install gpu requirements by default if --use-cpu is not used
+ Fix improved mask + color correction
+ Remove javascript, use https://github.com/w-e-w/sdwebui-close-confirmation-dialogue.git instead to prevent gradio from closing.

# 1.2.1 :

Add GPU support option : see https://github.com/glucauze/sd-webui-faceswaplab/pull/24

# 1.2.0 :

This version changes quite a few things.

+ The upscaled inswapper options are now moved to each face unit. This makes it possible to fine-tune the settings for each face.

+ Upscaled inswapper configuration in sd now concerns default values in each unit's interface.

+ Pre- and post-inpainting is now possible for each face. Here too, default options are set in the main sd settings.

+ Codeformer is no longer the default in post-processing. Don't be surprised if you get bad results by default. You can set it to default in the application's global settings

Bug fixes :

+ The problem of saving the grid should be solved.
+ The downscaling problem for inpainting should be solved.
+ Change model download logic and add checksum. This should prevent some bugs.

In terms of the API, it is now possible to create a remote checkpoint and use it in units. See the example in client_api or the tests in the tests directory.

See https://github.com/glucauze/sd-webui-faceswaplab/pull/19

# 1.1.2 :

+ Switch face checkpoint format from pkl to safetensors

See https://github.com/glucauze/sd-webui-faceswaplab/pull/4

## 1.1.1 :

+ Add settings for default inpainting prompts
+ Add better api support
+ Add api tests
+ bug fixes (extract, upscaling)
+ improve code checking and formatting (black, mypy, and pre-commit hooks)


## 1.1.0 :

All listed in features

+ add inpainting model selection => allow to select a different model for face inpainting
+ add source faces selection => allow to select the reference face if multiple face are present in reference image
+ add select by size => sort faces by size from larger to smaller
+ add batch option => allow to process images without txt2img or i2i in tabs
+ add segmentation mask for upscaled inpainter (based on codeformer implementation) : avoid square mask and prevent degradation of non-face parts of the image.

## 0.1.0 :

### Major :
+ add multiple face support
+ add face blending support (will blend sources faces)
+ add face similarity evaluation (will compare face to a reference)
    + add filters to discard images that are not rated similar enough to reference image and source images
+ add face tools tab
    + face extraction tool
    + face builder tool : will build a face model that can be reused
+ add faces models

### Minor :

Improve performance by not reprocessing source face each time

### Breaking changes

base64 and api not supported anymore (will be reintroduced in the future)