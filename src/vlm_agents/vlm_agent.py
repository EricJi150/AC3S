"""The Qwen3-VL handle shared by every agent.

One VLMAgent owns one model on one GPU. `self.lock` serializes every call, so
the A2/A3/A4 thread pool overlaps scheduling only, never GPU work.

The module-level `agent` is set once at startup via `set_agent()`. LangGraph
nodes take only `state`, leaving no way to pass the model in as an argument,
so the agent functions read it from here.
"""
import gc
import threading
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

MAX_IMAGE_EDGE = 1024
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

agent: Optional["VLMAgent"] = None


def set_agent(new_agent: Optional["VLMAgent"]) -> None:
    """Install the process-wide VLMAgent that the agent functions will use."""
    global agent
    agent = new_agent


def get_agent() -> "VLMAgent":
    """The installed VLMAgent, or a clear error if startup forgot to set one."""
    if agent is None:
        raise RuntimeError(
            "No VLMAgent installed. Call vlm_agents.set_agent(...) "
            "before invoking the workflow."
        )
    return agent


def _load_downsized(path: str) -> Image.Image:
    """Open an image as RGB, shrinking it so its longest edge is <= MAX_IMAGE_EDGE."""
    image = Image.open(path).convert("RGB")
    if max(image.size) > MAX_IMAGE_EDGE:
        ratio = MAX_IMAGE_EDGE / max(image.size)
        image = image.resize(tuple(int(d * ratio) for d in image.size), Image.LANCZOS)
    return image


class VLMAgent:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, use_quantization=True, device_id=None):
        self.lock = threading.Lock()
        device_map = {"": f"cuda:{device_id}"} if device_id is not None else "auto"
        if use_quantization:
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, quantization_config=qconfig, device_map=device_map,
                trust_remote_code=True, low_cpu_mem_usage=True
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=device_map,
                trust_remote_code=True, low_cpu_mem_usage=True
            )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def analyze_image(self, image_path: str, prompt: str,
                      max_new_tokens: int = 512, enable_thinking: bool = True) -> str:
        with self.lock:
            self.clear_memory()
            image = _load_downsized(image_path)
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=False)

            input_len = int(inputs['input_ids'].shape[1])
            new_token_ids = output_ids[:, input_len:]
            response = self.processor.batch_decode(new_token_ids, skip_special_tokens=True)[0]

            del inputs, output_ids, image
            self.clear_memory()
            return response.strip()

    def analyze_two_images(self, path1: str, path2: str, prompt: str,
                           max_new_tokens: int = 512, enable_thinking: bool = True) -> str:
        with self.lock:
            self.clear_memory()
            img1 = Image.open(path1).convert('RGB')
            img2 = Image.open(path2).convert('RGB')
            # NOTE: thumbnail() here, resize() in analyze_image. thumbnail is
            # in-place and preserves aspect ratio by fitting inside the box,
            # so the two paths can differ by a pixel on non-square inputs.
            # Kept as-is: changing it would change generated captions.
            for img in [img1, img2]:
                if max(img.size) > MAX_IMAGE_EDGE:
                    ratio = MAX_IMAGE_EDGE / max(img.size)
                    img.thumbnail(tuple(int(d * ratio) for d in img.size), Image.LANCZOS)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img1}, {"type": "image", "image": img2}, {"type": "text", "text": prompt}
            ]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            inputs = self.processor(text=[text], images=[img1, img2], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=False)

            input_len = int(inputs['input_ids'].shape[1])
            new_token_ids = output_ids[:, input_len:]
            response = self.processor.batch_decode(new_token_ids, skip_special_tokens=True)[0]

            del inputs, output_ids, img1, img2
            self.clear_memory()
            return response.strip()

    def analyze_text(self, prompt: str) -> str:
        # enable_thinking is intentionally not exposed here (relies on the chat
        # template's default), unlike the image-taking methods above which let
        # the caller set it explicitly. If you change that, check whether it
        # shifts this method's output distribution relative to the others.
        with self.lock:
            self.clear_memory()
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=False)

            input_len = int(inputs['input_ids'].shape[1])
            new_token_ids = output_ids[:, input_len:]
            response = self.processor.batch_decode(new_token_ids, skip_special_tokens=True)[0]

            del inputs, output_ids
            self.clear_memory()
            return response.strip()

    def analyze_batched(self, items: List[dict], max_new_tokens: int = 512,
                        enable_thinking: bool = True) -> List[str]:
        """Run a homogeneous-image-count batch through one model.generate.

        Each item dict should have:
          {"image_paths": [str, ...],   # 0..N image paths (homogeneous count
                                         #  across the batch is REQUIRED)
           "prompt":      str}

        Returns a list of response strings, one per item in input order.

        REQUIRES homogeneous image counts across the batch — caller's job.
        Mixing different image counts within a batch is not supported by
        the Qwen3-VL processor.
        """
        if not items:
            return []
        n_images_per = [len(it.get("image_paths", [])) for it in items]
        if len(set(n_images_per)) != 1:
            raise ValueError(
                f"analyze_batched requires homogeneous image counts; got {n_images_per}"
            )
        n_images = n_images_per[0]

        with self.lock:
            self.clear_memory()
            all_images = []
            messages_list = []
            for it in items:
                imgs = [_load_downsized(p) for p in it["image_paths"]]
                all_images.extend(imgs)
                content = [{"type": "image", "image": im} for im in imgs]
                content.append({"type": "text", "text": it["prompt"]})
                messages_list.append([{"role": "user", "content": content}])

            texts = [
                self.processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                ) for m in messages_list
            ]

            # Qwen3-VL is decoder-only, so batched generate needs left-padding:
            # with right-padding, sequences that finish early don't reliably
            # signal EOS and run all the way to max_new_tokens instead.
            # Save/restore so this doesn't leak into other callers.
            tok = getattr(self.processor, "tokenizer", None)
            saved_padding_side = None
            if tok is not None and hasattr(tok, "padding_side"):
                saved_padding_side = tok.padding_side
                tok.padding_side = "left"

            try:
                if n_images > 0:
                    inputs = self.processor(
                        text=texts, images=all_images, padding=True, return_tensors="pt"
                    )
                else:
                    inputs = self.processor(
                        text=texts, padding=True, return_tensors="pt"
                    )
            finally:
                if saved_padding_side is not None:
                    tok.padding_side = saved_padding_side

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = int(inputs["input_ids"].shape[1])

            # Passing eos_token_id explicitly lets batched generate stop each
            # sequence at its own EOS instead of waiting for the slowest one.
            eos_id = None
            if tok is not None and getattr(tok, "eos_token_id", None) is not None:
                eos_id = tok.eos_token_id

            with torch.no_grad():
                gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=0.1, do_sample=False)
                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            new_token_ids = output_ids[:, input_len:]
            responses = self.processor.batch_decode(new_token_ids, skip_special_tokens=True)

            del inputs, output_ids
            self.clear_memory()

        return [r.strip() for r in responses]
