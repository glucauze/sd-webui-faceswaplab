---
layout: page
title: FAQ
permalink: /faq/
---

Our issue tracker often contains requests that may originate from a misunderstanding of the software's functionality. We aim to address these queries; however, due to time constraints, we may not be able to respond to each request individually. This FAQ section serves as a preliminary source of information for commonly raised concerns. We recommend reviewing these before submitting an issue.

#### Improving Quality of Results

To get better quality results:

1. Ensure that the "Restore Face" option is enabled.
2. Consider using the "Upscaler" option. For finer control, you can use an upscaler from the "Extras" tab.
3. Use img2img with the denoise parameter set to `0.1`. Gradually increase this parameter until you achieve a balance of quality and resemblance.

You can also use the uspcaled inswapper. I mainly use it with the following options :

![](/assets/images/inswapper_options.png)


#### Replacing Specific Faces

If an image contains multiple faces and you want to swap specific ones: Use the "Comma separated face number(s)" option to select the face numbers you wish to swap.

#### Issues with Face Swapping

If a face did not get swapped, please check the following:

1. Ensure that the "Enable" option has been checked.
2. If you've ensured the above and your console doesn't show any errors, it means that the FaceSwapLab was unable to detect a face in your image or the image was detected as NSFW (Not Safe For Work).

#### Controversy Surrounding NSFW Content Filtering

We understand that some users might wish to have the option to disable content filtering, particularly for Not Safe for Work (NSFW) content. However, it's important to clarify our stance on this matter. We are not categorically against NSFW content. The concern arises specifically when the software is used to superimpose the face of a real person onto NSFW content.

If it were reliably possible to ensure that the faces being swapped were synthetic and not tied to real individuals, the inclusion of NSFW content would pose less of an ethical dilemma. However, in the absence of such a guarantee, making this feature easily accessible could potentially lead to misuse, which is an ethically risky scenario.

This is not our intention to impose our moral perspectives. Our goal is to comply with the requirements of the models used in the software and establish a balanced boundary that respects individual privacy and prevents potential misuse.

Requests to provide an option to disable the content filter will not be considered.

#### What is the role of the segmentation mask for the upscaled swapper?

The segmentation mask for the upscaled swapper is designed to avoid the square mask and prevent degradation of the non-face parts of the image. It is based on the Codeformer implementation. If "Use improved segmented mask (use pastenet to mask only the face)" and "upscaled inswapper" are checked in the settings, the mask will only cover the face, and will not be squared. However, depending on the image, this might introduce different types of problems such as artifacts on the border of the face.

#### How to increase speed of upscaled inswapper?

It is possible to choose LANCZOS for speed if Codeformer is enabled in the upscaled inswapper. The result is generally satisfactory.

#### Sharpening and color correction in upscaled swapper :

Sharpening can provide more natural results, but it may also add artifacts. The same goes for color correction. By default, these options are set to False.

#### I don't see any extension after restart

If you do not see any extensions after restarting, it is likely due to missing requirements, particularly if you're using Windows. Follow the instructions below:

1. Verify that there are no error messages in the terminal.
2. Double-check the Installation section of this document to ensure all the steps have been followed.

If you are running a specific configuration (for example, Python 3.11), please test the extension with a clean installation of the stable version of Diffusion before reporting an issue. This can help isolate whether the problem is related to your specific configuration or a broader issue with the extension.

#### Understanding Quality of Results

The model used in this extension initially reduces the resolution of the target face before generating a 128x128 image. This means that regardless of the original image's size, the resolution of the processed faces will not exceed 128x128. Consequently, this lower resolution might lead to quality limitations in the results.

The output of this process might not meet high expectations, but the use of the face restorer and upscaler can help improve these results to some extent.

The quality of results is inherently tied to the capabilities of the model and cannot be enhanced beyond its design. FaceSwapLab merely provides an interface for the underlying model. Therefore, unless the model from insighface is retrained and necessary alterations are made in the library (see below), the resulting quality may not meet high expectations.

Consider this extension as a low-cost alternative to more sophisticated tools like Lora, or as an addition to such tools. It's important to **maintain realistic expectations of the results** provided by this extension.


#### Issue: Incorrect Gender Detection

The gender detection functionality is handled by the underlying analysis model. As such, there might be instances where the detected gender may not be accurate. This is a limitation of the model and we currently do not have a way to improve this accuracy from our end.

#### Why isn't GPU support included?

While implementing GPU support may seem straightforward, simply requiring a modification to the onnxruntime implementation and a change in providers in the swapper, there are reasons we haven't included it as a standard option.

The primary consideration is the substantial VRAM usage of the SD models. Integrating the model on the GPU doesn't result in significant performance gains with the current state of the software. Moreover, the GPU support becomes truly beneficial when processing large numbers of frames or video. However, our experience indicates that this tends to cause more issues than it resolves.

Consequently, requests for GPU support as a standard feature will not be considered.

#### What is the 'Upscaled Inswapper' Option in SD FaceSwapLab?

The 'Upscaled Inswapper' is an option in SD FaceSwapLab which allows for upscaling of each face using an upscaller prior to its integration into the image. This is achieved by modifying a small segment of the InsightFace code.

The purpose of this feature is to enhance the quality of the face in the final image. While this process might slightly increase the processing time, it can deliver improved results. In certain cases, this could even eliminate the need for additional tools such as Codeformer or GFPGAN in postprocessing.

#### What is Face Blending?

Insighface works by creating an embedding for each face. An embedding is essentially a condensed representation of the facial characteristics.

The face blending process allows for the averaging of multiple face embeddings to generate a blended or composite face.

The benefits of face blending include:

+ Generation of a high-quality embedding based on multiple faces, thereby improving the face's representative accuracy.
+ Creation of a composite face that includes features from multiple individuals, which can be useful for diverse face recognition scenarios.

To create a composite face, two methods are available:

1. Use the Checkpoint Builder: This tool allows you to save a set of face embeddings that can be loaded later to create a blended face.
2. Use Image Batch Sources: By dropping several images into this tool, you can generate a blended face based on the faces in the provided images.

#### What is a face checkpoint?

A face checkpoint is a saved embedding of a face, generated from multiple images. This is accomplished via the build tool located in the `sd` tab. The build tool blends all images dropped into the tab and saves the resulting embedding to a file.

The primary advantage of face checkpoints is their size. An embedding is only around 2KB, meaning it's lightweight and can be reused later without requiring additional calculations.

Face checkpoints are saved as `.pkl` files. Please be aware that exchanging `.pkl` files carries potential security risks. These files, by default, are not secure and could potentially execute malicious code when opened. Therefore, extreme caution should be exercised when sharing or receiving this type of file.

#### How is similarity determined?

The similarity between faces is established by comparing their embeddings. In this context, a score of 1 signifies that the two faces are identical, while a score of 0 indicates that the faces are different.

You can remove images from the results if the generated image does not match the reference. This is done by adjusting the sliders in the "Faces" tab.

#### Which model is used?

The model employed here is based on InsightFace's "InSwapper". For more specific information, you can refer [here](https://github.com/deepinsight/insightface/blob/fc622003d5410a64c96024563d7a093b2a55487c/python-package/insightface/model_zoo/inswapper.py#L12).

This model was temporarily made public by the InsightFace team for research purposes. They have not provided any details about the training methodology.

The model generates faces with a resolution of 128x128, which is relatively low. For better results, the generated faces need to be upscaled. The InsightFace code is not designed for higher resolutions (see the [Router](https://github.com/deepinsight/insightface/blob/fc622003d5410a64c96024563d7a093b2a55487c/python-package/insightface/model_zoo/model_zoo.py#L35) class for more information).

#### Why not use SimSwap?

SimSwap models are based on older InsightFace architectures, and SimSwap has not been released as a Python package. Its incorporation would complicate the process, and it does not guarantee any substantial gain.

If you manage to implement SimSwap successfully, feel free to submit a pull request.
