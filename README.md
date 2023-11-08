### ComfyUI Bmad Nodes

Miscellaneous assortment of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

The nature of the nodes is varied, and they do not provide a comprehensive solution for any particular kind of application. 
The nodes can be roughly categorized in the following way:

- api: to help setup api requests (barebones). 
- computer vision: mainly for masking and collage purposes.
- general utility: simplify the workflow setup or implement some missing basic functionality.

______________________

<details><summary>
Documentation
</summary>

In order to keep the documentation brief and to the point, I will use the following icons for special nodes.
- ❔ the node has additional options when right-clicking, some of these options need to be used for the node to work.
- 📓 the node depends on an external library, and the requirements must be installed for it to work.
- 📄 the node relies on custom nodes external to this collection, they will only work if the needed nodes are installed.
- ❌the node won't work on vanilla comfyUI at the time of writing.
- ⚠️the node is potentially dangerous. Although they should be fairly safe in most cases, it is **NOT** advised to run them from unknown sources
unless you know what they are doing. For better visibility these nodes are forcefully painted white.


Furthermore, I won't provide any documentation for api nodes, as I think there are better, more comprehensive and already
documented, solutions available.
  
### General Purpose 


| Node                                     | Description                                                                                                                                                                                                                                                                                               |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| String                                   | Just a string (text). In case you want it written before connecting to a node or if some custom node does not work properly with the PrimitiveNode.                                                                                                                                                       |
| Add String to Many ❔                     | Will append/prepend the string `to_add` to all the other strings.                                                                                                                                                                                                                                         |
| Color Clip                               | Clips the `color` (or all the other colors) from an image. Both the target color or the complement can be set to white, black or remain untouched.                                                                                                                                                        |
| Color Clip ADE20k 📓️                    | Similar to Color Clip, but you pick the color from the ADE20k class list. Only useful for ADE20k semantic segmented images.                                                                                                                                                                               |
| MonoMerge                                | Selects the maximum (or minimum) value between two images. Mainly used for mask composition.                                                                                                                                                                                                              |
| AdjustRect                               | Receives a rectangle and returns a new rectangle that shares the same center but with width adjusted to a multiple of `xm` and height to a multiple of `ym`. Setting `round_mode` to **exact** will return a rectangle with the exact defined dimensions.                                                 |
| Repeat Into Grid                         | Tiles the provided image/latent into a grid of `columns`x`rows` tiles.                                                                                                                                                                                                                                    |
| Conditioning Grid (cond) ❔               | Creates conditioning areas of size `width`x`height`, forming a grid of `columns`x`rows` conditioning areas. The inputs notation can be read as: r{row}_c{column}. `strength` is the strength to by applied in all the areas, and `base` is the base conditioning prior to setting the tiles conditioning. |
| Conditioning Grid (string) ❔             | Similar to Conditioning Grid (cond), but generates the conditioning from the given strings (only).                                                                                                                                                                                                        |
| Conditioning Grid (string) Advanced 📄 ❔ | Similar to Conditioning Grid (string), but requires BlenderNeko's [Advanced CLIP Text Encode](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb).                                                                                                                                                       | 
| VAEEncodeBatch ❔                         | Receives multiples images and encodes them into a latent batch.                                                                                                                                                                                                                                           | 
| AnyToAny ❌ ⚠️                            | Can be used to convert data between different formats or compute stuff. The input data can be used in the expression using the letter `v`.                                                                                                                                                                |
| CLIPEncodeMultiple ❔                     | Receives individual strings → CLIPEncodes each → returns conditioning list.                                                                                                                                                                                                                               |
| CLIPEncodeMultipleAdvanced 📄 ❔          | Same as CLIPEncodeMultiple, but using BlenderNeko's [Advanced CLIP Text Encode](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb).                                                                                                                                                                     |
| ControlNetHadamard                       | Receives a list of conditionings and a list of images → Applies contronet only once per conditioning/image pair (does not apply every image to every conditioning).                                                                                                                                        |
| ControlNetHadamard (manual) ❔            | Similar to ControlNetHadamard but images are set via individual inputs.                                                                                                                                                                                                                                   |
| ToCondList ❔                             | Receives individual conditionings → returns a list with all the input conditionings.                                                                                                                                                                                                                      |
| ToLatentList ❔                             | Receives individual latents → returns a list with all the input latents.                                                                                                                                                                                                                      |
| ToImageList ❔                             | Receives individual images → returns a list with all the input images.                                                                                                                                                                                                                      |
| FromListGetConds ❔                             | Receives a list of conditionings → returns the conditionings via individual slots.                                                                                                                                                                                                                       |
| FromListGetLatents ❔                             | Receives a list of latents → returns the latents via individual slots.                                                                                                                                                                                                                       |
| FromListGetImages ❔                             | Receives a list of images → returns the images via individual slots.                                                                                                                                                                                                                       |


### CV (Computer Vision) nodes 

Nodes under the CV separator use or expose openCV functionalities.

I will only provide partial documentation here, to clarify how to use the more complex nodes.
The remaining nodes usage should be clear given the nodes' names.


#### Framed Mask Grab Cut

Returns a mask, in image format, with the result of the [grabcut](https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html).

<details>
<summary>
usage
</summary>

The `tresh` input should be a gray image, possibly a mask in black and white but not necessarily (read thresholds).
It is used to set most of the grabcut input mask's flags, excluding `GC_BGD` (sure background) which are set by the "frame". 

The "frame" - border margins of the image - has its size defined via the `pixels` input, and won't affect sides set to 
be ignored by the `frame_option` input (the corners common to neighbor sides will still be painted on the ignored sides).

The threshold inputs indicate the intensity threshold's used to set `GC_PR_FGD` (probable foreground) or `GC_FGD` (foreground).
The values **equal or above** the thresholds are set with the indicated flag. They can be setup in the following manners:
- To only use probable foreground, set threshold_FGD to **exactly 0**, and it will be ignored.
- To only use foreground, set threshold_FGD to a **lower value** than threshold_PR_FGD.
- To have both, keep threshold_FGD higher than threshold_PR_FGD (make sure your thresh input image contains
 values in the intended range).

The thresholds also work as safeguards against potential misleading or inconsistent input images,
where the image may appear to be only black and white, but actually contains values besides 0s and 255s.

##### Framed Mask Grab Cut 2

Similar to Framed Mask Grab Cut, but uses `thresh_maybe` to set the probable foreground, and `thresh_sure` to set the foreground. 
The `threshold` value is the same for both thresh image inputs; the `GC_FGD` flags are set by the `thresh_sure` on top of the `GC_PR_FGD` flags set by `thresh_maybe`.

</details>


#### Filter Contour  ⚠️

Will output contours depending on their fitness, where the fitness function must be provided
within the node's text box.

The expression may be long but can't have multiple instructions, only a single line that
returns the fitness when evaluated.

<details>
<summary>
usage
</summary>

`Select` argument options:
- `MAX`, `MIN`: select the contour (singular) with higher and lower fitness respectively, the evaluated expression should result in a **number**. 
- `FILTER`: filters the contours (plural) that satisfy the fitness condition, the evaluated expression should result in a **boolean**.
- `MODE`: selects the contour (singular) whose fitness score is the mode of all the contours fitness scores. 

To compute the fitness, the input parameters can be used with the following names:
- `c`: the contour being evaluated, from input contours
- `i`: input image (optional)
- `a`: input auxiliary contour (optional)

Functions from the math, opencv and numpy modules can be used with the prefixes: `m`; `cv`; and `np`, respectively.
Additionally, functions listed below can also be used without a prefix. 


The following is an example fitness function to get the contour that best matches the auxiliary contour (the lower the value, the better the match):
```
cv.matchShapes(c,a,1,0.0)
```

List of available functions:

- aspect_ratio(contour): bounding rectangle's width divided by height
- extent(contour): contour area divided by bounding rect area
- solidity(contour): contour area divided by hull area
- equi_diameter(contour): how round is the shape `math.sqrt(4 * area / math.pi)`
- center(contour)
- contour_mask(contour, image)
- mean_color(contour, image)
- mean_intensity(contour, image)
- extreme_points(contour)
- intercepts_mask(contour, image)  `does not cache result`

All the listed functions cache the results at least once (details vary); they don't create computational overhead 
for being called more than once.
This behavior was also added to the following list of opencv functions, which must be called **without the cv prefix**:

- boundingRect
- contourArea
- arcLength (called without the boolean arg; is always sent with `true`)
- minEnclosingRect
- minEnclosingCircle
- fitEllipse
- convexHull

</details>

</details>