import os
import sys
from typing import Literal, Dict, Optional
import time
import fire


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.png"),
    model_id_or_path: str ="KBlueLeaf/kohaku-v2.1", #"mirroring/pastel-mix", #"black-forest-labs/FLUX.1-dev",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    width: int = 1024,
    height: int = 1024,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = False,
    seed: int = int(time.time()),
):
    
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    output : str, optional
        The output image file to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default False.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """
    init = time.time()
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,    
        t_index_list=[0, 16, 32, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        mode="txt2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type="none",
        seed=seed,
    )

    stre = time.time()
    stream.prepare(
        prompt=prompt,
        negative_prompt="blurry",
        num_inference_steps=50,
    )
    for _ in range(stream.batch_size - 1):
        stream()
    curr = time.time()
    print(f"total time: {curr- init} seconds, stream time: {curr-stre}")
    output_image = stream()
    output_image.save(output)


if __name__ == "__main__":
    fire.Fire(main)
