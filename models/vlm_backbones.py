"""
Thin wrappers around Qwen2.5-VL-7B-Instruct and LLaVA-Video-7B-Qwen2 that expose
a single generate(video_path, prompt) -> str interface, so the eval loop doesn't
need to know which model it's talking to.

Both models are used zero-shot here (no fine-tuning) as the "semantic scoring"
baseline discussed in Related Work section 2 -- the point is to see how far
language-grounded instruction tuning alone gets you, with no cortical signal
involved at all.

Decoding: both scorers default to sampling (temperature=0.5) rather than
greedy decoding. Greedy decoding collapses to whatever token has the single
highest logit every time, which for genuinely ambiguous rating items (e.g.
color_comfort, valence) can produce the exact same rating on every video --
not because the model "has no opinion," but because it's a decoding
artifact. Sampling lets the model's actual uncertainty show up as variance
instead. Pass temperature=0.0 to get the old greedy behavior back.

GeminiScorer is a third option, calling the Gemini API rather than running a
local model -- no GPU, no CUDA/transformers version conflicts, no llava-env
vs qwen-env split. It has its own module docstring below with the specifics
that don't apply to the other two (file upload/caching, JSON-mode output,
retry handling).
"""

import os
import time

import numpy as np
import torch


# --------------------------------------------------------------------------
# Shared frame sampling: both models take a list of frames rather than a raw
# video stream in the way most eval scripts use them, so we sample uniformly
# and let each wrapper decide the frame count.
# --------------------------------------------------------------------------
def sample_frame_indices(n_frames, num_samples):
    if n_frames <= num_samples:
        return list(range(n_frames))
    idx = np.linspace(0, n_frames - 1, num_samples)
    return sorted(set(int(round(i)) for i in idx))


class Qwen25VLScorer:
    """
    Zero-shot scorer using Qwen2.5-VL-7B-Instruct.
    Model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    Requires: pip install transformers>=4.49 qwen-vl-utils accelerate
    """

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda",
                 num_frames=16, max_new_tokens=512, dtype=torch.bfloat16,
                 temperature=0.5, top_p=0.9, do_sample=None):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.device = device
        self.num_frames = num_frames
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        # do_sample defaults to "sample iff temperature > 0"; can be forced
        # either way, but temperature=0.0 is the intended way to get plain
        # greedy decoding back for a sanity-check run.
        self.do_sample = do_sample if do_sample is not None else (temperature > 0)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def generate(self, video_path, prompt):
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "nframes": self.num_frames,
                },
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response.strip()


class LLaVAVideoScorer:
    """
    Zero-shot scorer using LLaVA-Video-7B-Qwen2 (lmms-lab).
    Model card: https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2
    Requires the LLaVA-NeXT codebase (this checkpoint isn't yet loadable via
    plain `transformers`):
        pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
    """
 
    def __init__(self, model_id="lmms-lab/LLaVA-Video-7B-Qwen2", device="cuda",
                 num_frames=64, max_new_tokens=512, conv_template="qwen_1_5",
                 temperature=0.5, top_p=0.9, do_sample=None,
                 attn_implementation="sdpa"):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
 
        self.device = device
        self.num_frames = num_frames
        self.max_new_tokens = max_new_tokens
        self.conv_template = conv_template
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample if do_sample is not None else (temperature > 0)
 
        model_name = get_model_name_from_path(model_id)
        # LLaVA-NeXT's builder defaults to attn_implementation="flash_attention_2",
        # which raises ImportError if the flash_attn package isn't installed.
        # "sdpa" uses PyTorch's built-in scaled_dot_product_attention instead --
        # no extra install needed, slightly slower than flash-attn but otherwise
        # numerically equivalent. Pass attn_implementation="flash_attention_2"
        # here if flash_attn is installed and you want the speed.
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_id, None, model_name, torch_dtype="bfloat16", device_map=device,
            attn_implementation=attn_implementation,
        )
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.image_processor = image_processor
 
    def _load_video_frames(self, video_path):
        # decord is the loader LLaVA-Video's own eval scripts use
        from decord import VideoReader, cpu
 
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_idx = sample_frame_indices(len(vr), self.num_frames)
        frames = vr.get_batch(frame_idx).asnumpy()  # (T, H, W, 3)
        return frames
 
    @torch.inference_mode()
    def generate(self, video_path, prompt):
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
 
        frames = self._load_video_frames(video_path)
        video_tensor = self.image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"].to(self.device, dtype=torch.bfloat16)
 
        full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
 
        input_ids = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
 
        gen_kwargs = {
            "images": [video_tensor],
            "modalities": ["video"],
            "max_new_tokens": self.max_new_tokens,
            "stopping_criteria": [stopping_criteria],
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
 
        output_ids = self.model.generate(input_ids, **gen_kwargs)
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response.strip()

class GeminiScorer:
    """
    Zero-shot scorer calling the Gemini API (e.g. gemini-3.1-flash-lite).
    Unlike Qwen2.5-VL and LLaVA-Video, this makes real network calls and
    costs real (if small) money per call -- no local GPU or model weights
    involved, so none of the CUDA/transformers-version issues from the other
    two scorers apply here.

    Requires: pip install google-genai
    Requires: export GEMINI_API_KEY=... (or pass api_key= directly)

    Video files are uploaded via Gemini's File API and CACHED by path for
    the lifetime of this scorer instance. Your eval loop reuses the same
    scorer across every split x run combination, and the same physical
    video often appears across multiple runs (same split, different --run)
    -- caching avoids re-uploading that file every time. Gemini-hosted files
    expire after 48h; a full split x run sweep should comfortably finish
    inside that window, so no persistence beyond the process lifetime is
    implemented -- a fresh script run starts with a cold cache and
    re-uploads, which is fine.

    Uses response_mime_type="application/json" to get a JSON-mode response
    from the API itself, rather than relying on prompts.py's regex-repair
    fallback the way the local models sometimes need to -- Gemini won't
    produce free text wrapped around the JSON the way Qwen/LLaVA sometimes
    do, though a response can still get cut short if max_new_tokens is set
    too low, same as the other two scorers.
    """

    def __init__(self, model_id="gemini-3.1-flash-lite", api_key=None,
                 temperature=0.5, top_p=0.9, max_new_tokens=512,
                 max_retries=5, retry_base_delay=120.0,
                 num_frames=None, do_sample=None):
        from google import genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment "
                "variable, or pass api_key= explicitly to GeminiScorer."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._file_cache = {}  # video_path -> uploaded google.genai file object

        # num_frames/do_sample accepted-but-unused: eval_vlm_baseline.py
        # conditionally passes num_frames through scorer_kwargs, and would
        # raise a TypeError here if this constructor didn't accept it. Gemini
        # samples video at 1 FPS server-side by default rather than a fixed
        # frame count; fine-grained control exists via videoMetadata.fps on
        # the uploaded file but isn't wired up here.
        if num_frames is not None:
            print(f"[GeminiScorer] num_frames={num_frames} was passed but is not used -- "
                  f"Gemini samples video server-side (1 FPS by default), not by frame count.")

    def _call_with_retry(self, fn, description):
        """Shared retry-with-backoff wrapper for any single Gemini API call
        (upload, poll, or generate) -- transient server errors (like a bare
        500 INTERNAL) can land on ANY of these, not just generate_content, so
        every network call goes through this rather than only the final
        generation step. Without this, a single unlucky 500 during file
        upload or ACTIVE-state polling crashes the whole sweep instead of
        just retrying, which is what was happening before this fix."""
        from google.genai import errors

        last_error = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except (errors.APIError, RuntimeError) as e:
                last_error = e
                delay = self.retry_base_delay * (2 ** attempt)
                print(f"[GeminiScorer] {description} failed on attempt "
                      f"{attempt + 1}/{self.max_retries}: {e!r} -- retrying in {delay:.1f}s")
                time.sleep(delay)

        raise RuntimeError(
            f"Gemini {description} failed after {self.max_retries} attempts: {last_error!r}"
        )

    def _get_uploaded_file(self, video_path):
        if video_path in self._file_cache:
            return self._file_cache[video_path]

        uploaded = self._call_with_retry(
            lambda: self.client.files.upload(file=video_path),
            f"file upload for {video_path}",
        )
        # Gemini processes an uploaded video asynchronously; it must reach
        # ACTIVE state before it can be referenced in a generate_content
        # call, or the request will fail outright. Each individual poll is
        # its own retried call -- a transient error on poll #3 shouldn't
        # throw away the (successful, already-paid-for) upload from poll #1.
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = self._call_with_retry(
                lambda: self.client.files.get(name=uploaded.name),
                f"file status poll for {video_path}",
            )
        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(
                f"Gemini file upload for {video_path} ended in state "
                f"{uploaded.state.name!r}, expected ACTIVE"
            )

        self._file_cache[video_path] = uploaded
        return uploaded

    def generate(self, video_path, prompt):
        from google.genai import types

        uploaded_file = self._get_uploaded_file(video_path)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_new_tokens,
            response_mime_type="application/json",
        )

        def _do_generate():
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[uploaded_file, prompt],
                config=config,
            )
            if response.text is None:
                # Can happen if generation was cut off by a safety filter or
                # hit max_output_tokens with nothing usable yet -- treat as a
                # retryable failure rather than crashing on None.strip().
                raise RuntimeError(f"Gemini returned no text (finish_reason: "
                                    f"{response.candidates[0].finish_reason if response.candidates else 'unknown'})")
            return response.text.strip()

        return self._call_with_retry(_do_generate, f"generate_content for {video_path}")
    
def load_scorer(name, **kwargs):
    if name == "qwen":
        return Qwen25VLScorer(**kwargs)
    elif name == "llava":
        return LLaVAVideoScorer(**kwargs)
    elif name == "gemini":
        return GeminiScorer(**kwargs)
    raise ValueError(f"Unknown model name: {name}")