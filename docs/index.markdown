---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

FaceSwapLab is an extension for Stable Diffusion that simplifies the use of [insighface models](https://insightface.ai/) for face-swapping. It has evolved from sd-webui-faceswap and some part of sd-webui-roop. However, a substantial amount of the code has been rewritten to improve performance and to better manage masks.

Some key [features](features) include the ability to reuse faces via checkpoints, multiple face units, batch process images, sort faces based on size or gender, and support for vladmantic. It also provides a face inpainting feature.

![](/assets/images/main_interface.png)

Link to github repo : [https://github.com/glucauze/sd-webui-faceswaplab](https://github.com/glucauze/sd-webui-faceswaplab)

While FaceSwapLab is still under development, it has reached a good level of stability. This makes it a reliable tool for those who are interested in face-swapping within the Stable Diffusion environment. As with all projects of this type, it's expected to improve and evolve over time.


## Disclaimer and license

In short:

+ **Ethical Guideline:** This extension should not be forked to create a public, easy way to circumvent NSFW filtering.
+ **License:** This software is distributed under the terms of the GNU Affero General Public License (AGPL), version 3 or later.
+ **Model License:** This software uses InsightFace's pre-trained models, which are available for non-commercial research purposes only.

### Ethical Guideline

This extension is **not intended to facilitate the creation of not safe for work (NSFW) or non-consensual deepfake content**. Its purpose is to bring consistency to image creation, making it easier to repair existing images, or bring characters back to life.

While the code for this extension is licensed under the AGPL in compliance with models and other source materials, it's important to stress that **we strongly discourage any attempts to fork this project to create an uncensored version**. Any modifications to the code to enable the production of such content would be contrary to the ethical guidelines we advocate for.

We will comply with European regulations regarding this type of software. As required by law, the code may include both visible and invisible watermarks. If your local laws prohibit the use of this extension, you should not use it.

From an ethical perspective, the main goal of this extension is to generate consistent images by swapping faces. It's important to note that we've done our best to integrate censorship features. However, when users can access the source code, they might bypass these censorship measures. That's why we urge users to use this extension responsibly and avoid any malicious use. We emphasize the importance of respecting people's privacy and consent when swapping faces in images. We discourage any activities that could harm others, invade their privacy, or negatively affect their well-being.

Additionally, we believe it's important to make the public aware of these tools and the ease with which deepfakes can be created. As technology improves, we need to be more critical and skeptical when we encounter media content. By promoting media literacy, we can reduce the negative impact of misusing these tools and encourage responsible use in the digital world.

### Software License

This software is distributed under the terms of the GNU Affero General Public License (AGPL), version 3 or later. It is provided "AS IS", without any express or implied warranties, including but not limited to the implied warranties of merchantability and fitness for a particular purpose. In no event shall the author be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage. Users are encouraged to review the license in full and to use the software in accordance with its terms.

If any user violates their country's legal and ethical rules, we don't accept any liability for this code repository.

### Models License

This software utilizes the pre-trained models `buffalo_l` and `inswapper_128.onnx`, which are provided by InsightFace. These models are included under the following conditions:

_InsightFace's pre-trained models are available for non-commercial research purposes only. This includes both auto-downloading models and manually downloaded models._ from [insighface licence](https://github.com/deepinsight/insightface/tree/master/python-package)

Users of this software must strictly adhere to these conditions of use. The developers and maintainers of this software are not responsible for any misuse of InsightFace's pre-trained models.

Please note that if you intend to use this software for any commercial purposes, you will need to train your own models or find models that can be used commercially.

## Acknowledgments

This project contains code adapted from the following sources:
+ codeformer : https://github.com/sczhou/CodeFormer
+ PSFRGAN : https://github.com/chaofengc/PSFRGAN
+ insightface : https://insightface.ai/
+ ifnude : https://github.com/s0md3v/ifnude
+ sd-webui-roop : https://github.com/s0md3v/sd-webui-roop

## Alternatives

+ https://github.com/idinkov/sd-deepface-1111
+ https://github.com/s0md3v/sd-webui-roop
