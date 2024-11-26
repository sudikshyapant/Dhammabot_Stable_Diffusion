# Dhammabot
This repository contains code for leveraging Stable Diffusion, a generative AI model, to generate high-quality images based on textual prompts. The project utilizes the Diffusers library, Transformers, and SciPy for efficient implementation.

# Installation
To run the code, make sure you have the required packages installed. You can install them using pip:

```
!pip install --upgrade diffusers transformers scipy
!pip install torchvision
```

# Usage

## Import the necessary libraries and modules:

```python

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
```

## Load the Stable Diffusion model and scheduler:
```python
model_id = "CompVis/stable-diffusion-v1-4"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
```

## Define a prompt and generate an image:
```python

prompt = "A comic strip where a man uses Buddhist mindfulness meditation to help deal with stress and becomes happy again."
image = pipe(prompt).images[0]
image.save("Dhamma.png")
```
## Display the generated image:
```python
display(image)
```
# Additional Notes
This project leverages the Stable Diffusion model for high-fidelity image generation from textual descriptions.
Make sure to run the code on a GPU-enabled environment for optimal performance.
Experiment with different prompts and explore the capabilities of Stable Diffusion in generating diverse visual content.

Feel free to reach out for any questions or feedback regarding the project. Happy coding!
