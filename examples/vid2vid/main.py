import av  # Add this import at the top with other imports
import os,time
import sys
from typing import Literal, Dict, Optional
from fractions import Fraction
import numpy as np
import fire
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    input: str,
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.mp4"),
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog ears, thick frame glasses",
    scale: float = 1.0,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    enable_similar_image_filter: bool = True,
    seed: int = 2,
):

    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    input : str, optional
        The input video name to load images from.
    output : str, optional
        The output video name to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    scale : float, optional
        The scale of the image, by default 1.0.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """
    init = time.time()
    video_info = read_video(input)
    video = video_info[0] / 255
    fps = video_info[2]["video_fps"]
    height = int(video.shape[1] * scale)
    width = int(video.shape[2] * scale)

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id,
        lora_dict=lora_dict,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=False,
        mode="img2img",
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.9,
        use_denoising_batch=use_denoising_batch,
        seed=seed,
        cfg_type="initialize"
    )
    stre = time.time()
    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    video_result = torch.zeros(video.shape[0], height, width, 3)

    for _ in range(stream.batch_size):
        stream(image=video[0].permute(2, 0, 1))

    for i in tqdm(range(video.shape[0])):
        output_image = stream(video[i].permute(2, 0, 1))
        video_result[i] = output_image.permute(1, 2, 0)

    # video_result = video_result * 255
    # write_video(output, video_result[2:], fps=fps)

    # Convert video to proper format
    video_result = (video_result * 255).to(torch.uint8)
    video_result = video_result.cpu().numpy()[2:]  # Convert to numpy and skip first 2 frames

    # Write video using av
    container = av.open(output, mode='w')
    stream = container.add_stream('h264', rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    for frame_data in video_result:
        frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
        packet = stream.encode(frame)
        container.mux(packet)

    # Flush encoder
    packet = stream.encode(None)
    container.mux(packet)
    container.close()
    curr = time.time()
    print(f"total time: {curr- init} seconds, stream time: {curr-stre}")

if __name__ == "__main__":
    fire.Fire(main)
