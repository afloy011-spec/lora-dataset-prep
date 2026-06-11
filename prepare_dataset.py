#!/usr/bin/env python3
"""
prepare_dataset.py v2.2.0
LoRA dataset preparation tool -- Ostris AI Toolkit / Z-Image Turbo

Prepares raw images into a clean numbered dataset with .txt caption sidecars,
README.md, and metadata.json. Designed for character LoRA first; supports
style, object, clothing, and environment LoRA types.

Usage (interactive):
    python prepare_dataset.py

Usage (CLI, dry-run -- default):
    python prepare_dataset.py --source ./raw --output ./my_dataset --type character --trigger mychar01

Usage (with VLM auto-captioning):
    python prepare_dataset.py --source ./raw --output ./my_dataset \\
        --type character --trigger mychar01 --captions vlm --execute

Usage (with resize and repeats):
    python prepare_dataset.py --source ./raw --output ./my_dataset \\
        --type character --trigger mychar01 --resize 1024 --repeats 10 --execute

Usage (validate an existing dataset, read-only):
    python prepare_dataset.py --validate-only --output ./my_dataset --trigger mychar01

Dry-run is always printed first. Changes only happen when --execute is passed.

Optional dependencies:
    pip install Pillow          # WebP conversion, resize, image quality checks
    pip install anthropic       # VLM auto-captioning (--captions vlm)
"""

import argparse
import asyncio
import base64
import hashlib
import json
import re
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

# -- Version --------------------------------------------------------------------
VERSION = "2.2.0"

# -- Supported formats ----------------------------------------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
CONVERT_EXTS = {".webp"}
# -- Caption limits -------------------------------------------------------------
CAPTION_WORD_TARGET_MIN = 15
CAPTION_WORD_MAX = 35
CAPTION_WORD_LONG = 60

# -- Attribute-sticking detection -----------------------------------------------
# A phrase appearing in this fraction of captions is flagged as a sticking risk
# (the skill rule: a repeated adjective/attribute binds to the trigger).
STICKING_PHRASE_FRACTION = 0.5
# Words ignored when detecting repeated attributes: function words, the subject
# nouns, and structural/shot-type words (shot variety is reported separately).
STICKING_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "to", "over",
    "behind", "near", "from", "his", "her", "their", "is", "are", "by", "for",
    "man", "woman", "person", "shot", "medium", "close", "up", "full", "body",
    "profile", "view", "wearing", "standing", "sitting", "look", "looking",
}
# Tokens that signal an intentional media-type anchor (meant to repeat).
STICKING_MEDIA_TOKENS = {"still", "tv", "show", "screencap", "photograph", "photo", "sitcom"}

# -- Image quality thresholds ---------------------------------------------------
MIN_IMAGE_DIM = 512
BLUR_THRESHOLD = 100.0
NEAR_DUPE_THRESHOLD = 10
VLM_CONCURRENCY = 5
LOCAL_VLM_MODEL = "Salesforce/blip-image-captioning-large"

# -- LoRA types -----------------------------------------------------------------
LORA_TYPES = ["character", "style", "object", "clothing", "environment"]

# -- Caption templates per LoRA type --------------------------------------------

CAPTION_TEMPLATES = {
    "character": [
        "{trigger}, [MEDIA_TYPE: a photograph/a still from a TV show], medium shot of [SUBJECT: a man/a woman] wearing [CLOTHING], standing with hands at sides, neutral expression, front view, plain studio background, soft studio lighting",
        "{trigger}, [MEDIA_TYPE], close-up of [SUBJECT] wearing [CLOTHING], looking at the camera, calm expression, front view, plain background, soft daylight",
        "{trigger}, [MEDIA_TYPE], full body shot of [SUBJECT] wearing [CLOTHING], standing with hands at sides, neutral expression, front view, minimal background, natural light",
        "{trigger}, [MEDIA_TYPE], medium shot of [SUBJECT] wearing [CLOTHING], sitting on a chair, slight smile, three quarter view, simple room background, warm indoor lighting",
        "{trigger}, [MEDIA_TYPE], close-up of [SUBJECT] wearing [CLOTHING], looking slightly to the side, neutral expression, side view, plain background, soft window light",
        "{trigger}, [MEDIA_TYPE], full body shot of [SUBJECT] wearing [CLOTHING], walking, neutral expression, side view, street background, overcast outdoor lighting",
        "{trigger}, [MEDIA_TYPE], medium shot of [SUBJECT] wearing [CLOTHING], arms crossed, serious expression, front view, plain background, soft studio lighting",
        "{trigger}, [MEDIA_TYPE], full body shot of [SUBJECT] wearing [CLOTHING], back to camera, neutral expression, rear view, outdoor background, natural daylight",
        "{trigger}, [MEDIA_TYPE], medium shot of [SUBJECT] wearing [CLOTHING], sitting cross-legged, relaxed expression, front view, floor background, warm indoor lighting",
        "{trigger}, [MEDIA_TYPE], close-up of [SUBJECT] wearing [CLOTHING], looking away, thoughtful expression, three quarter view, blurred background, soft daylight",
        "{trigger}, [MEDIA_TYPE], full body shot of [SUBJECT] wearing [CLOTHING], standing with one hand on hip, calm expression, three quarter view, minimal interior background, natural light",
        "{trigger}, [MEDIA_TYPE], medium shot of [SUBJECT] wearing [CLOTHING], standing outdoors, neutral expression, front view, outdoor background, overcast light",
        "{trigger}, [MEDIA_TYPE], full body shot of [SUBJECT] wearing [CLOTHING], seen from above, neutral expression, top-down view, clean floor background, soft diffused lighting",
        "{trigger}, [MEDIA_TYPE], medium shot of [SUBJECT] wearing [CLOTHING], leaning against a wall, casual expression, side view, wall background, indoor lighting",
        "{trigger}, [MEDIA_TYPE], profile view of [SUBJECT] wearing [CLOTHING], side profile portrait, neutral expression, plain background, window light",
        "{trigger}, [MEDIA_TYPE], full body shot of [SUBJECT] wearing [CLOTHING], dynamic pose, focused expression, three quarter view, plain background, studio lighting",
        "{trigger}, [MEDIA_TYPE], close-up of [SUBJECT] wearing [CLOTHING], looking up, surprised expression, low angle, dark background, dramatic lighting",
        "{trigger}, [MEDIA_TYPE], medium shot of [SUBJECT] wearing [CLOTHING], sitting at a table, focused expression, front view, table background, indoor lighting",
        "{trigger}, [MEDIA_TYPE], full body shot of [SUBJECT] wearing [CLOTHING], standing outdoors, relaxed expression, front view, outdoor background, golden hour light",
        "{trigger}, [MEDIA_TYPE], close-up of [SUBJECT] wearing [CLOTHING], smiling, front view, plain background, soft studio lighting",
    ],
    "style": [
        "{trigger}, [CONTENT: a person/landscape/object], [STYLE HINT: flat colors/soft lines/hatching]",
        "{trigger}, [CONTENT], [MOOD: vibrant/muted/warm], [COMPOSITION: centered/rule of thirds]",
        "{trigger}, [SUBJECT], [COLOR PALETTE: pastel/monochrome/bold], [RENDERING: illustrated/painterly]",
        "{trigger}, [SCENE], [TEXTURE: rough/smooth/textured brushwork], [LIGHTING STYLE: diffused/hard]",
        "{trigger}, [CONTENT], [STYLISTIC ELEMENT], [ATMOSPHERE: cozy/dramatic/serene]",
    ],
    "object": [
        "{trigger}, a [OBJECT CLASS], front view, on white background",
        "{trigger}, a [OBJECT CLASS], side view, on white background",
        "{trigger}, a [OBJECT CLASS], three quarter view, product photo",
        "{trigger}, a [OBJECT CLASS], top-down view, flat lay",
        "{trigger}, a [OBJECT CLASS], close-up of [DETAIL], detail shot",
        "{trigger}, a [OBJECT CLASS], back view, isolated",
        "{trigger}, a [OBJECT CLASS], in use, [CONTEXT]",
        "{trigger}, a [OBJECT CLASS], low angle view, dramatic lighting",
        "{trigger}, a [OBJECT CLASS], on [SURFACE: wood table/marble/fabric], lifestyle",
        "{trigger}, a [OBJECT CLASS], multiple angles, studio light",
    ],
    "clothing": [
        "{trigger}, [CLOTHING ITEM], worn by a [woman/man], front view, full body, plain background",
        "{trigger}, [CLOTHING ITEM], worn by a [woman/man], side view, medium shot",
        "{trigger}, [CLOTHING ITEM], flat lay, overhead, white background",
        "{trigger}, [CLOTHING ITEM], detail close-up, [TEXTURE/PATTERN]",
        "{trigger}, [CLOTHING ITEM], worn by a [woman/man], back view, full body",
        "{trigger}, [CLOTHING ITEM], worn by a [woman/man], walking, street background",
        "{trigger}, [CLOTHING ITEM], on hanger, studio background",
        "{trigger}, [CLOTHING ITEM], detail of [ZIPPER/BUTTONS/STITCHING], close-up",
    ],
    "environment": [
        "{trigger}, wide establishing shot, [TIME OF DAY: morning/golden hour/night], overcast",
        "{trigger}, interior view, [ROOM/SPACE], [LIGHTING: natural/warm/fluorescent]",
        "{trigger}, detail shot, [ARCHITECTURAL ELEMENT or TEXTURE]",
        "{trigger}, aerial overview, [LOCATION], [WEATHER]",
        "{trigger}, ground level, [SCENE], [MOOD: dramatic/serene/urban]",
        "{trigger}, entrance or transition point, [DESCRIPTION]",
        "{trigger}, [LOCATION], night, artificial lighting",
        "{trigger}, [LOCATION], close-up texture detail",
        "{trigger}, panoramic view, [LOCATION], [TIME OF DAY]",
        "{trigger}, [LOCATION], different season or light condition",
    ],
}

MINIMAL_CAPTIONS = {
    "character": "{trigger}, portrait photo",
    "style": "{trigger}",
    "object": "{trigger}, product photo",
    "clothing": "{trigger}, clothing item",
    "environment": "{trigger}, establishing shot",
}

# -- Caption strategy notes -----------------------------------------------------

CAPTION_STRATEGY_NOTES = {
    "character": (
        "Formula: [trigger], [media type], [shot type] of a [man/woman], [clothing], [pose/action], "
        "[expression], [background/setting], [lighting]. Natural English, not tag soup. "
        "Describe variable attributes; omit identity traits (face, eyes, skin, hair unless changeable). "
        "Caption length: 15-35 words. No quality tags. No poetic language."
    ),
    "style": (
        "Captions describe the CONTENT (what is depicted), not the style. The LoRA learns the "
        "style itself; describing style keywords in captions can cause style bleed. "
        "Add only minimal, non-specific style hints if needed. "
        "Dataset should have diverse subjects all rendered in the target style."
    ),
    "object": (
        "Captions include the trigger token AND the object class (e.g. 'a white running shoe'). "
        "This anchors the semantic category and prevents confusion. "
        "Cover many angles, lighting scenarios, and usage contexts."
    ),
    "clothing": (
        "Captions describe the garment clearly (type, style, color, texture if visible). "
        "Include the trigger token. Show the clothing from many angles and contexts. "
        "Flat-lay and on-body shots both help. Describe the wearer generically."
    ),
    "environment": (
        "Captions describe the varying elements (lighting, time, detail) while the location "
        "itself is the constant the LoRA learns. Wide + detail shots, different lighting "
        "conditions, and angles are important for scene generalization."
    ),
}

# -- VLM prompts per LoRA type --------------------------------------------------

VLM_PROMPTS = {
    "character": (
        "Write a caption for character LoRA training (Z-Image Turbo). "
        "Formula: {trigger}, [media type], [shot type] of a [man/woman], [clothing], [pose/action], "
        "[expression], [background/setting], [lighting]. "
        "Start with exactly '{trigger}'. Use natural English sentences, NOT comma-separated tags. "
        "Include: media type (e.g. 'a photograph', 'a still from a TV show'), shot type, clothing, "
        "pose, facial expression, camera angle, background, lighting. "
        "Do NOT describe identity traits: face shape, eye color, skin tone, hair (unless a removable accessory). "
        "No quality tags (masterpiece, 8k). 15-35 words. Return ONLY the caption text, nothing else."
    ),
    "style": (
        "Describe the CONTENT of this image for style LoRA training. "
        "Start with exactly '{trigger}'. Describe what is depicted (subject, composition, colors), "
        "not the artistic style itself. The LoRA learns style from images. "
        "8-30 words. Return ONLY the caption."
    ),
    "object": (
        "Describe this product/object photo for object LoRA training. "
        "Start with exactly '{trigger}', then the object class. "
        "Include: view angle, notable details, surface/background. "
        "8-30 words. Return ONLY the caption."
    ),
    "clothing": (
        "Describe this clothing item for clothing LoRA training. "
        "Start with exactly '{trigger}'. Include: garment type, color, material/texture if visible, "
        "view angle, presentation (worn/flat lay/hanger). Describe the wearer generically if present. "
        "8-30 words. Return ONLY the caption."
    ),
    "environment": (
        "Describe this scene/environment for environment LoRA training. "
        "Start with exactly '{trigger}'. Include: shot type, time of day, lighting, notable features. "
        "The location is the constant the LoRA learns; describe what varies. "
        "8-30 words. Return ONLY the caption."
    ),
}

# -- Optional dependencies ------------------------------------------------------

def has_pillow() -> bool:
    try:
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def has_vlm_support() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def has_local_vlm_support() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


# -- Helpers --------------------------------------------------------------------

def is_safe_filename(name: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._\-]+$', name))


def word_count(text: str) -> int:
    return len(text.split())


# -- Image quality helpers ------------------------------------------------------

def get_image_dimensions(path: Path) -> tuple:
    from PIL import Image
    with Image.open(path) as img:
        return img.size


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_blur_score(path: Path) -> float:
    from PIL import Image, ImageFilter
    with Image.open(path) as img:
        thumb = img.copy()
        thumb.thumbnail((256, 256))
        gray = thumb.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        pixels = list(edges.tobytes())
        if len(pixels) < 2:
            return 0.0
        return statistics.variance(pixels)


def compute_perceptual_hash(path: Path, hash_size: int = 9) -> str:
    from PIL import Image
    with Image.open(path) as img:
        small = img.resize((hash_size, hash_size - 1), Image.LANCZOS).convert("L")
        pixels = list(small.tobytes())
        w = hash_size
        bits = []
        for y in range(hash_size - 1):
            for x in range(w - 1):
                idx = y * w + x
                bits.append("1" if pixels[idx] < pixels[idx + 1] else "0")
        bit_str = "".join(bits)
        return hex(int(bit_str, 2))[2:].zfill(len(bit_str) // 4)


def hamming_distance(hash1: str, hash2: str) -> int:
    b1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    b2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))


def find_exact_duplicates(images: list) -> list:
    hash_map = {}
    for img_path in images:
        h = compute_file_hash(img_path)
        hash_map.setdefault(h, []).append(img_path)
    return [group for group in hash_map.values() if len(group) > 1]


def find_near_duplicates(images: list, threshold: int = NEAR_DUPE_THRESHOLD) -> list:
    if not has_pillow():
        return []
    hashes = []
    for img_path in images:
        try:
            hashes.append((img_path, compute_perceptual_hash(img_path)))
        except Exception:
            pass

    pairs = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hamming_distance(hashes[i][1], hashes[j][1])
            if 0 < dist <= threshold:
                pairs.append((hashes[i][0], hashes[j][0], dist))
    return pairs


def check_images_quality(images: list, blur_threshold: float = BLUR_THRESHOLD) -> dict:
    issues = {"too_small": [], "blurry": [], "exact_duplicates": [], "near_duplicates": []}

    pillow_ok = has_pillow()

    for img_path in images:
        if pillow_ok:
            try:
                w, h = get_image_dimensions(img_path)
                if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                    issues["too_small"].append((img_path, w, h))
            except Exception:
                pass
            if blur_threshold > 0:
                try:
                    score = detect_blur_score(img_path)
                    if score < blur_threshold:
                        issues["blurry"].append((img_path, round(score, 1)))
                except Exception:
                    pass

    issues["exact_duplicates"] = find_exact_duplicates(images)
    issues["near_duplicates"] = find_near_duplicates(images)
    return issues


# -- Image processing -----------------------------------------------------------

def convert_webp_to_png(src: Path, dst: Path) -> None:
    from PIL import Image
    with Image.open(src) as img:
        img.save(dst, "PNG")


def resize_image(src: Path, dst: Path, target_size: int) -> bool:
    from PIL import Image
    with Image.open(src) as img:
        w, h = img.size
        if min(w, h) <= target_size:
            if src != dst:
                shutil.copy2(src, dst)
            return False
        if w < h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        fmt = "PNG" if dst.suffix.lower() == ".png" else "JPEG"
        save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
        resized.save(dst, fmt, **save_kwargs)
        return True


# -- VLM captioning -------------------------------------------------------------

async def _caption_one_async(sem, async_client, image_path: Path, lora_type: str,
                             trigger: str, model: str, index: int) -> tuple:
    """Returns (index, caption, error). Exactly one of caption/error is set,
    so failures stay attributed to the image that actually failed."""
    async with sem:
        try:
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            ext = image_path.suffix.lower()
            media_types = {
                ".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".webp": "image/webp",
            }
            media_type = media_types.get(ext, "image/png")
            prompt = VLM_PROMPTS[lora_type].format(trigger=trigger)

            message = await async_client.messages.create(
                model=model,
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

            caption = message.content[0].text.strip().strip('"\'')
            if not caption.lower().startswith(trigger.lower()):
                caption = f"{trigger}, {caption}"
            return index, caption, None
        except Exception as e:
            return index, None, str(e)


def caption_batch_vlm(ops: list, lora_type: str, trigger: str,
                      model: str = "claude-sonnet-4-6",
                      concurrency: int = VLM_CONCURRENCY) -> dict:
    import anthropic

    async def _run():
        async_client = anthropic.AsyncAnthropic()
        sem = asyncio.Semaphore(concurrency)
        tasks = []
        for op in ops:
            tasks.append(_caption_one_async(
                sem, async_client, op["src"], lora_type, trigger, model, op["index"]
            ))
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())

    captions = {}
    errors = {}
    for idx, caption, error in results:
        if error is not None:
            errors[idx] = error
        else:
            captions[idx] = caption

    return {"captions": captions, "errors": errors}


# Filler phrases plain caption models (BLIP) emit that violate the caption
# formula. Stripped from the front of local captions before the trigger is added.
LOCAL_FILLER_PREFIXES = [
    "there is a", "there is an", "there are", "there's a", "there's an",
    "this is an image of a", "this is an image of an", "this is an image of",
    "this is a picture of a", "this is a picture of an", "this is a picture of",
    "this is a photo of a", "this is a photo of an", "this is a photo of",
    "this is a", "this is an",
    "an image of a", "an image of an", "an image of",
    "a picture of a", "a picture of an", "a picture of",
    "a photo of a", "a photo of an", "a photo of",
    "a close up of a", "a close up of an", "a close-up of a", "a close-up of an",
    "image of a", "picture of a", "photo of a",
]


def clean_local_caption(text: str) -> str:
    """Strip leading filler phrases from plain (BLIP-style) caption output.

    BLIP emits things like 'there is a man in a kitchen' or 'this is an image
    of a man'. These break the caption formula and waste words, so we trim the
    filler and keep the descriptive remainder.
    """
    cleaned = text.strip()
    lowered = cleaned.lower()
    for prefix in sorted(LOCAL_FILLER_PREFIXES, key=len, reverse=True):
        if lowered.startswith(prefix + " "):
            remainder = cleaned[len(prefix):].lstrip()
            # Re-insert an article so the sentence still reads naturally.
            if not re.match(r"^(a|an|the)\b", remainder, re.IGNORECASE):
                remainder = "a " + remainder
            cleaned = remainder
            break
    return cleaned.strip()


def caption_batch_local(ops: list, lora_type: str, trigger: str,
                        model_name: str = LOCAL_VLM_MODEL) -> dict:
    from PIL import Image as PILImage
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_florence = "florence" in model_name.lower()

    print(f"  Loading {model_name} on {device}...")

    if is_florence:
        from transformers import AutoProcessor, AutoModelForCausalLM
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    captions = {}
    errors = {}
    total = len(ops)

    for op in ops:
        try:
            image = PILImage.open(op["src"]).convert("RGB")

            if is_florence:
                task = "<MORE_DETAILED_CAPTION>"
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
                inputs = processor(text=task, images=image, return_tensors="pt").to(device, torch_dtype)
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=100, do_sample=False, num_beams=3,
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(
                    generated_text, task=task, image_size=(image.width, image.height)
                )
                text = parsed.get(task, "").strip()
            else:
                inputs = processor(images=image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=100, num_beams=3)
                text = processor.decode(out[0], skip_special_tokens=True).strip()

            if text:
                if not is_florence:
                    text = clean_local_caption(text)
                text = text[0].lower() + text[1:] if text and text[0].isupper() else text
                caption = f"{trigger}, {text}"
            else:
                caption = f"{trigger}, portrait photo"

            captions[op["index"]] = caption
            print(f"  [{op['index']}/{total}] {op['src'].name} -> OK")

        except Exception as e:
            errors[op["index"]] = str(e)
            print(f"  [{op['index']}/{total}] {op['src'].name} -> ERROR: {e}")

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"captions": captions, "errors": errors}


# -- Source discovery -----------------------------------------------------------

def discover_images(source_dir: Path) -> dict:
    result = {"supported": [], "webp": [], "unsupported": [], "all": []}
    for f in sorted(source_dir.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in SUPPORTED_EXTS:
            result["supported"].append(f)
            result["all"].append(f)
        elif ext in CONVERT_EXTS:
            result["webp"].append(f)
            result["all"].append(f)
        elif ext not in {".txt", ".json", ".md", ".yaml", ".yml"}:
            result["unsupported"].append(f)
    return result


# -- Caption generation ---------------------------------------------------------

def get_caption(index: int, lora_type: str, trigger: str, mode: str) -> str:
    if mode == "minimal":
        return MINIMAL_CAPTIONS[lora_type].format(trigger=trigger)

    if mode in ("vlm", "local"):
        return f"[auto-caption -- generated during execution]"

    templates = CAPTION_TEMPLATES[lora_type]
    template = templates[index % len(templates)]
    return template.format(trigger=trigger)


# -- Padding --------------------------------------------------------------------

def pad_number(n: int, total: int) -> str:
    if total >= 100:
        return f"{n:03d}"
    return f"{n:02d}"


# -- Planning -------------------------------------------------------------------

def build_plan(images: list, opts: dict) -> list:
    total = len(images)
    ops = []
    for i, src in enumerate(images, start=1):
        num = pad_number(i, total)
        src_ext = src.suffix.lower()

        if src_ext in SUPPORTED_EXTS:
            out_ext = src_ext
            needs_convert = False
        else:
            out_ext = ".png"
            needs_convert = True

        out_name = num + out_ext
        txt_name = num + ".txt"
        caption = get_caption(i - 1, opts["lora_type"], opts["trigger"], opts["captions"])

        ops.append({
            "index": i,
            "src": src,
            "out_image": opts["train_dir"] / out_name,
            "out_txt": opts["train_dir"] / txt_name,
            "out_name": out_name,
            "txt_name": txt_name,
            "needs_convert": needs_convert,
            "caption": caption,
            "action": "move" if opts["move"] else "copy",
        })

    return ops


def print_plan(ops: list, opts: dict, quality: dict = None) -> None:
    print()
    print("=" * 60)
    print("  DRY RUN -- planned operations")
    print("=" * 60)
    print(f"  Source:         {opts['source_dir']}")
    print(f"  Output:         {opts['output_dir']}")
    print(f"  Train folder:   {opts['train_dir'].name}/")
    print(f"  LoRA type:      {opts['lora_type']}")
    print(f"  Trigger token:  {opts['trigger']}")
    print(f"  Action:         {'move' if opts['move'] else 'copy'}")
    print(f"  Captions:       {opts['captions']}")
    if opts.get("resize"):
        print(f"  Resize:         shortest side -> {opts['resize']}px")
    if opts.get("repeats"):
        print(f"  Repeats:        {opts['repeats']}x (folder: {opts['train_dir'].name}/)")
    print(f"  Backup raw:     {'yes' if opts['backup'] else 'no'}")
    print(f"  Images found:   {len(ops)}")

    if quality:
        has_issues = (quality["too_small"] or quality["blurry"]
                      or quality["exact_duplicates"] or quality["near_duplicates"])
        if has_issues:
            print()
            print("  IMAGE QUALITY WARNINGS:")
            for path, w, h in quality["too_small"]:
                print(f"    SMALL  {path.name} ({w}x{h} -- min {MIN_IMAGE_DIM}px)")
            for path, score in quality["blurry"]:
                print(f"    BLUR   {path.name} (score {score} -- threshold {BLUR_THRESHOLD})")
            for group in quality["exact_duplicates"]:
                names = ", ".join(f.name for f in group)
                print(f"    DUPE   {names} (identical)")
            for path_a, path_b, dist in quality["near_duplicates"]:
                print(f"    NEAR   {path_a.name} ~ {path_b.name} (distance {dist}/{NEAR_DUPE_THRESHOLD})")

    print()
    print("  Files that will be created:")
    for op in ops:
        conv = " [webp-->png]" if op["needs_convert"] else ""
        resize_tag = f" [resize->{opts['resize']}px]" if opts.get("resize") else ""
        ellipsis = "..." if len(op['caption']) > 60 else ""
        print(f"    {op['out_name']}{conv}{resize_tag}  <--  {op['src'].name}")
        print(f"    {op['txt_name']}  -->  \"{op['caption'][:60]}{ellipsis}\"")
    print()
    print(f"  README.md      will be created")
    print(f"  metadata.json  will be created")
    if opts["backup"]:
        print(f"  raw/           originals will be copied")
    print()
    print("  To apply: run again with --execute")
    print("=" * 60)


# -- Execution ------------------------------------------------------------------

def execute_plan(ops: list, opts: dict) -> dict:
    output_dir: Path = opts["output_dir"]
    train_dir: Path = opts["train_dir"]
    raw_dir: Path = opts["raw_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    if opts["backup"]:
        raw_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "copied": 0,
        "moved": 0,
        "converted": 0,
        "resized": 0,
        "vlm_captioned": 0,
        "vlm_errors": 0,
        "captions_written": 0,
        "errors": [],
        "file_list": [],
    }

    total = len(ops)
    caption_mode = opts["captions"]
    do_resize = opts.get("resize")
    vlm_model = opts.get("vlm_model", "claude-sonnet-4-6")
    local_model = opts.get("local_model", LOCAL_VLM_MODEL)

    # -- Batch auto-captioning (before any file moves) --
    auto_captions = {}
    if caption_mode == "vlm":
        print(f"  Captioning {total} images via Claude API ({VLM_CONCURRENCY} concurrent)...")
        batch_result = caption_batch_vlm(ops, opts["lora_type"], opts["trigger"],
                                         vlm_model, VLM_CONCURRENCY)
        auto_captions = batch_result["captions"]
        batch_errors = batch_result["errors"]
        stats["vlm_captioned"] = len(auto_captions)
        stats["vlm_errors"] = len(batch_errors)
        print(f"  VLM done: {len(auto_captions)} OK, {len(batch_errors)} errors")
        for idx, err in batch_errors.items():
            src_name = next((op["src"].name for op in ops if op["index"] == idx), f"#{idx}")
            print(f"  WARN  VLM failed for {src_name}: {err}")
    elif caption_mode == "local":
        is_florence = "florence" in str(local_model).lower()
        print(f"  Captioning {total} images via {local_model} (local)...")
        if not is_florence:
            print("  NOTE  Plain caption models (BLIP) do NOT follow the caption formula.")
            print("        Output is generic (e.g. 'a man in a kitchen') and must be")
            print("        rewritten before training. Prefer --captions vlm or a Florence-2")
            print("        local model for character LoRA.")
        batch_result = caption_batch_local(ops, opts["lora_type"], opts["trigger"], local_model)
        auto_captions = batch_result["captions"]
        batch_errors = batch_result["errors"]
        stats["vlm_captioned"] = len(auto_captions)
        stats["vlm_errors"] = len(batch_errors)
        print(f"  Local captioning done: {len(auto_captions)} OK, {len(batch_errors)} errors")

    for op in ops:
        try:
            if opts["backup"]:
                shutil.copy2(op["src"], raw_dir / op["src"].name)

            # -- Image processing --
            was_resized = False
            if op["needs_convert"]:
                if not has_pillow():
                    msg = f"Pillow not installed -- cannot convert {op['src'].name}. Run: pip install Pillow"
                    stats["errors"].append(msg)
                    print(f"  ERROR: {msg}")
                    continue
                convert_webp_to_png(op["src"], op["out_image"])
                if do_resize:
                    was_resized = resize_image(op["out_image"], op["out_image"], do_resize)
                if opts["move"]:
                    op["src"].unlink()
                stats["converted"] += 1
            elif do_resize and has_pillow():
                was_resized = resize_image(op["src"], op["out_image"], do_resize)
                if opts["move"]:
                    op["src"].unlink()
                    stats["moved"] += 1
                else:
                    stats["copied"] += 1
            elif opts["move"]:
                shutil.move(op["src"], op["out_image"])
                stats["moved"] += 1
            else:
                shutil.copy2(op["src"], op["out_image"])
                stats["copied"] += 1

            if was_resized:
                stats["resized"] += 1

            # -- Resolve caption --
            caption = op["caption"]
            was_vlm = False
            if caption_mode in ("vlm", "local"):
                if op["index"] in auto_captions:
                    caption = auto_captions[op["index"]]
                    was_vlm = True
                else:
                    templates = CAPTION_TEMPLATES[opts["lora_type"]]
                    caption = templates[(op["index"] - 1) % len(templates)].format(trigger=opts["trigger"])
                    caption += "  # VLM_FAILED -- fill manually"

            op["caption"] = caption
            op["out_txt"].write_text(caption, encoding="utf-8")
            stats["captions_written"] += 1

            stats["file_list"].append({
                "image": op["out_name"],
                "caption_file": op["txt_name"],
                "caption": caption,
            })

            progress = f"[{op['index']}/{total}]"
            tags = ""
            if was_resized:
                tags += " [resized]"
            if was_vlm:
                tags += " [vlm]"
            print(f"  OK  {progress} {op['out_name']}{tags}  +  {op['txt_name']}")

        except Exception as e:
            msg = f"{op['src'].name}: {e}"
            stats["errors"].append(msg)
            print(f"  ERROR: {msg}")

    return stats


# -- README + metadata ----------------------------------------------------------

def write_readme(output_dir: Path, opts: dict, stats: dict) -> None:
    dataset_name = output_dir.name
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    image_count = stats.get("copied", 0) + stats.get("moved", 0) + stats.get("converted", 0)

    train_folder = opts["train_dir"].name

    lines = [
        f"# {dataset_name}",
        "",
        f"**Created:** {now}",
        "",
        "## Overview",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Dataset name | `{dataset_name}` |",
        f"| LoRA type | {opts['lora_type']} |",
        f"| Target training stack | Ostris AI Toolkit / Z-Image Turbo |",
        f"| Trigger token | `{opts['trigger']}` |",
        f"| Number of images | {image_count} |",
        f"| Caption mode | {opts['captions']} |",
        f"| Train folder | `{train_folder}/` |",
        f"| Naming convention | sequential zero-padded (01, 02 / 001, 002) |",
    ]

    if opts.get("resize"):
        lines.append(f"| Resize | shortest side -> {opts['resize']}px |")
    if opts.get("repeats"):
        lines.append(f"| Repeats | {opts['repeats']}x per epoch |")
    if stats.get("resized"):
        lines.append(f"| Images resized | {stats['resized']} |")
    if stats.get("vlm_captioned"):
        lines.append(f"| VLM-captioned | {stats['vlm_captioned']} |")

    lines += [
        "",
        "## File Structure",
        "",
        "```",
        f"{dataset_name}/",
        f"|---- {train_folder}/",
        "|   |---- 01.png",
        "|   |---- 01.txt",
        "|   |---- 02.png",
        "|   |---- 02.txt",
        "|   |---- ...",
        "|---- raw/          <-- original files (backup)",
        "|---- README.md     <-- this file",
        "|---- metadata.json",
        "```",
        "",
        "## Captioning Strategy",
        "",
        CAPTION_STRATEGY_NOTES[opts["lora_type"]],
        "",
    ]

    if opts["captions"] == "vlm":
        lines += [
            "## VLM Auto-Captioning",
            "",
            "Captions were generated automatically by Claude Vision API.",
            "Each image was analyzed and described according to the LoRA type rules.",
            "Review captions and adjust any inaccuracies before training.",
            "",
        ]

    lines += [
        "## Caption Rules",
        "",
        "- Every caption starts with the trigger token",
        f"- Caption length: {CAPTION_WORD_TARGET_MIN}-{CAPTION_WORD_MAX} words (flag above {CAPTION_WORD_LONG})",
        "- No quality tags (masterpiece, best quality, 8k, etc.)",
        "- No SD 1.5 tag-soup style",
        "- Short, natural English descriptions",
    ]

    if opts["captions"] == "template":
        lines.append("- Template captions contain `[BRACKETS]` -- fill these in before training")

    lines += [
        "",
        "## Naming Convention",
        "",
        "- Images: `01.png`, `02.png` (zero-padded; `001` if 100+ images)",
        "- Captions: `01.txt`, `02.txt` (same basename as image)",
        "- Dataset folder: lowercase, hyphens, no spaces or special chars",
        "- No Cyrillic, emojis, or spaces in any filename",
        "",
        "## Training Notes",
        "",
        f"- Trigger token: `{opts['trigger']}` -- add this to every inference prompt",
        "- Training adapter: `ostris/zimage_turbo_training_adapterV2` (remove at inference)",
        "- Recommended rank: 16 (32 if identity is weak)",
        "- Recommended steps: 2000-3000 for character LoRA",
        "- Caption dropout: 0.0 (Z-Image Turbo keeps captions sparse, not dropped)",
        "- Resolution: 1024",
        "- Remove adapter at inference -- load only the trained LoRA",
        "",
        "## Validation Warnings",
        "",
    ]

    warnings = stats.get("validation_warnings", [])
    if warnings:
        for w in warnings:
            lines.append(f"- WARNING: {w}")
    else:
        lines.append("- No warnings")

    lines += [
        "",
        f"*Generated by prepare_dataset.py v{VERSION}*",
    ]

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_metadata(output_dir: Path, opts: dict, stats: dict) -> None:
    image_count = stats.get("copied", 0) + stats.get("moved", 0) + stats.get("converted", 0)
    data = {
        "dataset_name": output_dir.name,
        "lora_type": opts["lora_type"],
        "target_training_stack": "Ostris AI Toolkit / Z-Image Turbo",
        "trigger_token": opts["trigger"],
        "created_at": datetime.now().isoformat(),
        "image_count": image_count,
        "supported_formats": sorted(SUPPORTED_EXTS),
        "caption_mode": opts["captions"],
        "train_folder": opts["train_dir"].name,
        "resize": opts.get("resize"),
        "repeats": opts.get("repeats"),
        "resized_count": stats.get("resized", 0),
        "vlm_captioned": stats.get("vlm_captioned", 0),
        "vlm_errors": stats.get("vlm_errors", 0),
        "files": stats.get("file_list", []),
        "validation_warnings": stats.get("validation_warnings", []),
        "errors": stats.get("errors", []),
        "notes": CAPTION_STRATEGY_NOTES[opts["lora_type"]],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# -- Validation -----------------------------------------------------------------

def repeated_phrase_warnings(captions: dict, fraction: float = STICKING_PHRASE_FRACTION,
                             max_warn: int = 6) -> list:
    """Flag content phrases that repeat across captions (attribute-sticking risk).

    Implements the skill rule "same adjective in every caption causes sticking".
    Structural/shot-type words and the intentional media-type anchor are ignored so
    only real attribute repeats (gaze, lighting, clothing colour) surface.
    """
    texts = [c for c in captions.values() if c]
    n = len(texts)
    if n < 4:
        return []

    phrase_docs: dict = {}
    for c in texts:
        words = re.findall(r"[a-z][a-z\-]+", c.lower())
        seen = set()
        for size in (2, 3):
            for i in range(len(words) - size + 1):
                gram = words[i:i + size]
                if all(w in STICKING_STOPWORDS for w in gram):
                    continue
                if any(w in STICKING_MEDIA_TOKENS for w in gram):
                    continue
                if gram[0] in STICKING_STOPWORDS and gram[-1] in STICKING_STOPWORDS:
                    continue
                seen.add(" ".join(gram))
        for p in seen:
            phrase_docs[p] = phrase_docs.get(p, 0) + 1

    hits = [(p, d, d / n) for p, d in phrase_docs.items() if d / n >= fraction]
    hits.sort(key=lambda x: (-x[1], -len(x[0])))

    kept: list = []
    for phrase, docs, frac in hits:
        if any(phrase in k and phrase != k for k, _, _ in kept):
            continue
        kept.append((phrase, docs, frac))

    return [
        f"Repeated phrase '{p}' in {d}/{n} captions ({f*100:.0f}%) -- vary it unless intentional (attribute sticking risk)"
        for p, d, f in kept[:max_warn]
    ]


def validate_dataset(train_dir: Path, trigger: str, run_quality: bool = True,
                     caption_mode: str = None,
                     blur_threshold: float = BLUR_THRESHOLD) -> list:
    warnings = []

    if not train_dir.exists():
        return ["train/ directory does not exist"]

    image_files = []
    image_bases = set()
    txt_bases = set()
    captions = {}

    for f in sorted(train_dir.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        base = f.stem
        if ext in SUPPORTED_EXTS:
            image_bases.add(base)
            image_files.append(f)
        elif ext == ".txt":
            txt_bases.add(base)
            try:
                text = f.read_text(encoding="utf-8").strip()
                captions[base] = text
            except Exception:
                warnings.append(f"Cannot read caption file: {f.name}")

    for base in sorted(image_bases):
        if base not in txt_bases:
            warnings.append(f"Missing caption file for image: {base}")

    for base in sorted(txt_bases):
        if base not in image_bases:
            warnings.append(f"Orphan caption: {base}.txt has no matching image")

    stem_counts: dict = {}
    for f in sorted(train_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            stem_counts.setdefault(f.stem, []).append(f.name)
    for stem, names in stem_counts.items():
        if len(names) > 1:
            warnings.append(f"Duplicate base name '{stem}': {', '.join(names)}")

    for base, text in sorted(captions.items()):
        if not text:
            warnings.append(f"Empty caption: {base}.txt")
            continue
        if trigger and trigger.lower() not in text.lower():
            warnings.append(f"Trigger token '{trigger}' missing in: {base}.txt")
        if not text.lower().startswith(trigger.lower()):
            warnings.append(f"Caption does not start with trigger token '{trigger}': {base}.txt")
        wc = word_count(text)
        if wc > CAPTION_WORD_LONG:
            warnings.append(f"Caption too long ({wc} words, max ~60 for training): {base}.txt -- \"{text[:60]}...\"")
        elif wc > CAPTION_WORD_MAX:
            warnings.append(f"Caption long ({wc} words, target 15-35): {base}.txt -- \"{text[:60]}...\"")
        elif caption_mode != "minimal" and wc < CAPTION_WORD_TARGET_MIN:
            warnings.append(f"Caption short ({wc} words, target 15-35): {base}.txt -- \"{text}\"")
        if "VLM_FAILED" in text:
            warnings.append(f"VLM captioning failed -- fill manually: {base}.txt")
        if "[" in text and "]" in text and re.search(r'\[[A-Z]', text):
            warnings.append(f"Unfilled [BRACKET] placeholder in: {base}.txt -- \"{text[:60]}...\"")
        # Filler phrases from plain caption models (BLIP) that break the formula.
        if re.search(r'(?:^|,\s*)(?:there (?:is|are)|there\'s|this is (?:a|an)|an? (?:image|picture|photo) of)\b',
                     text, re.IGNORECASE):
            warnings.append(f"Filler phrase (e.g. 'there is a') -- rewrite per formula: {base}.txt -- \"{text[:60]}...\"")
        # Multiple subjects: fatal for a single-subject character LoRA. Require an
        # actual person noun so phrases like "three quarter view" don't false-positive.
        if re.search(r'\b(?:a man and a woman|a woman and a man|a group of|'
                     r'another (?:man|woman|person|guy|boy|girl)|'
                     r'other (?:people|men|women|persons)|'
                     r'(?:two|three|four|five|several|multiple)\s+(?:\w+\s+){0,2}'
                     r'(?:men|women|people|persons|guys|boys|girls|kids|children|figures))\b',
                     text, re.IGNORECASE):
            warnings.append(f"Possible multiple subjects -- crop or relabel for single-subject LoRA: {base}.txt -- \"{text[:60]}...\"")

    if captions:
        caption_values = list(captions.values())
        unique_captions = set(caption_values)
        if len(unique_captions) == 1 and len(caption_values) > 1:
            warnings.append("All captions are identical -- consider adding variety")
        elif len(unique_captions) < len(caption_values) * 0.5 and len(caption_values) > 3:
            warnings.append(
                f"Low caption diversity: {len(unique_captions)} unique out of {len(caption_values)} -- review for duplicates"
            )
        warnings.extend(repeated_phrase_warnings(captions))

    for f in sorted(train_dir.iterdir()):
        if f.is_file() and not is_safe_filename(f.name):
            warnings.append(f"Unsafe filename (spaces/Cyrillic/special chars): {f.name}")

    # -- Image quality checks on final dataset --
    if run_quality and image_files and has_pillow():
        for img_path in image_files:
            try:
                w, h = get_image_dimensions(img_path)
                if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                    warnings.append(f"Image too small ({w}x{h}, min {MIN_IMAGE_DIM}px): {img_path.name}")
            except Exception:
                pass
            if blur_threshold > 0:
                try:
                    score = detect_blur_score(img_path)
                    if score < blur_threshold:
                        warnings.append(f"Image may be blurry (score {round(score, 1)}, threshold {blur_threshold}): {img_path.name}")
                except Exception:
                    pass

        for group in find_exact_duplicates(image_files):
            names = ", ".join(f.name for f in group)
            warnings.append(f"Duplicate images (identical content): {names}")
        for path_a, path_b, dist in find_near_duplicates(image_files):
            warnings.append(f"Near-duplicate images ({dist}/{NEAR_DUPE_THRESHOLD} distance): {path_a.name} ~ {path_b.name}")

    return warnings


# -- Interactive prompts --------------------------------------------------------

def ask(prompt: str, default: str = "", choices: list = None) -> str:
    choice_hint = ""
    if choices:
        choice_hint = f" [{'/'.join(choices)}]"
    default_hint = f" (default: {default})" if default else ""
    while True:
        val = input(f"{prompt}{choice_hint}{default_hint}: ").strip()
        if not val and default:
            return default
        if choices and val not in choices:
            print(f"  Please choose one of: {', '.join(choices)}")
            continue
        if val:
            return val
        print("  This field is required.")


def run_interactive(opts: dict) -> dict:
    print()
    print(f"prepare_dataset.py v{VERSION} -- LoRA Dataset Preparation Tool")
    print("-" * 60)
    print()

    if not opts.get("source_dir"):
        src = ask("Source folder (raw images)")
        opts["source_dir"] = Path(src)

    if not opts.get("output_dir"):
        out = ask("Output dataset folder")
        opts["output_dir"] = Path(out)

    if not opts.get("lora_type"):
        opts["lora_type"] = ask("LoRA type", default="character", choices=LORA_TYPES)

    if not opts.get("trigger"):
        opts["trigger"] = ask("Trigger token (e.g. mychar01, zbxobj)")

    if opts.get("captions") is None:
        caption_choices = ["template", "minimal"]
        if has_local_vlm_support():
            caption_choices.append("local")
        if has_vlm_support():
            caption_choices.append("vlm")
        if "local" not in caption_choices and "vlm" not in caption_choices:
            print("  (Auto-captioning unavailable -- install: pip install transformers torch)")
        opts["captions"] = ask("Caption mode", default="template", choices=caption_choices)

    if opts.get("move") is None:
        action = ask("Copy or move images?", default="copy", choices=["copy", "move"])
        opts["move"] = (action == "move")

    if opts.get("backup") is None:
        backup_ans = ask("Backup originals to raw/?", default="yes", choices=["yes", "no"])
        opts["backup"] = (backup_ans == "yes")

    if opts.get("resize") is None and has_pillow():
        resize_ans = ask("Resize images? (enter target px or 'no')", default="no")
        if resize_ans != "no":
            try:
                opts["resize"] = int(resize_ans)
            except ValueError:
                print("  Invalid number, skipping resize")
                opts["resize"] = None

    if opts.get("repeats") is None:
        repeats_ans = ask("Repeats per epoch? (number or 'no')", default="no")
        if repeats_ans != "no":
            try:
                opts["repeats"] = int(repeats_ans)
            except ValueError:
                print("  Invalid number, skipping repeats")
                opts["repeats"] = None

    return opts


# -- Argument parsing -----------------------------------------------------------

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="LoRA dataset preparation tool (Ostris AI Toolkit / Z-Image Turbo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", help="Source folder with raw images")
    parser.add_argument("--output", help="Output dataset folder (will be created)")
    parser.add_argument("--type", dest="lora_type", choices=LORA_TYPES, default=None,
                        help="LoRA type: character/style/object/clothing/environment")
    parser.add_argument("--trigger", help="Trigger token (e.g. mychar01)")
    parser.add_argument("--captions", choices=["minimal", "template", "vlm", "local"],
                        default=None,
                        help=(
                            "Caption mode: "
                            "minimal=trigger token only, "
                            "template=varied templates with [BRACKET] placeholders, "
                            "vlm=auto-caption via Claude API (pip install anthropic), "
                            "local=auto-caption via Florence-2 locally (pip install transformers torch)"
                        ))
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--copy", action="store_true", default=None,
                              help="Copy images (default; does not delete originals)")
    action_group.add_argument("--move", action="store_true", default=None,
                              help="Move images (deletes originals after copy)")
    parser.add_argument("--no-backup", action="store_true", default=False,
                        help="Skip creating raw/ backup folder")
    parser.add_argument("--convert-webp", action="store_true", default=False,
                        help="Convert WebP images to PNG (requires Pillow)")
    parser.add_argument("--resize", type=int, default=None, metavar="PX",
                        help="Resize images so shortest side equals PX (e.g. 1024). Requires Pillow")
    parser.add_argument("--repeats", type=int, default=None, metavar="N",
                        help="Create N_trigger/ folder structure instead of train/ (kohya/Ostris repeats)")
    parser.add_argument("--vlm-model", default="claude-sonnet-4-6", metavar="MODEL",
                        help="Model for VLM captioning (default: claude-sonnet-4-6)")
    parser.add_argument("--local-model", default=LOCAL_VLM_MODEL, metavar="MODEL",
                        help=f"Model for local captioning (default: {LOCAL_VLM_MODEL})")
    parser.add_argument("--no-quality-check", action="store_true", default=False,
                        help="Skip image quality checks (dimensions, blur, duplicates)")
    parser.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD, metavar="N",
                        help=f"Blur detection threshold (default: {BLUR_THRESHOLD}). "
                             "The detector is a heuristic -- smooth, well-lit portraits can "
                             "false-positive; lower the value or pass 0 to disable blur checks")
    parser.add_argument("--validate-only", action="store_true", default=False,
                        help="Only validate an existing dataset (no files are changed). "
                             "Use with --output (dataset root or train folder) and --trigger. "
                             "Trigger is read from metadata.json if omitted")
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Actually perform file operations (default is dry-run only)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    args = parser.parse_args()

    opts = {
        "source_dir": Path(args.source) if args.source else None,
        "output_dir": Path(args.output) if args.output else None,
        "lora_type": args.lora_type,
        "trigger": args.trigger,
        "captions": args.captions,
        # Keep None when neither --copy nor --move was passed so interactive
        # mode can ask; main() falls back to copy/backup defaults otherwise.
        "move": True if args.move else (False if args.copy else None),
        "backup": False if args.no_backup else None,
        "convert_webp": args.convert_webp,
        "resize": args.resize,
        "repeats": args.repeats,
        "vlm_model": args.vlm_model,
        "local_model": args.local_model,
        "no_quality_check": args.no_quality_check,
        "blur_threshold": args.blur_threshold,
        "validate_only": args.validate_only,
        "execute": args.execute,
    }
    return opts


# -- Validate-only mode -----------------------------------------------------------

def resolve_train_dir(root: Path, trigger: str = None, repeats: int = None) -> Path:
    """Find the train folder inside a dataset root.

    Accepts the dataset root (containing train/ or N_trigger/) or the train
    folder itself (a directory that directly contains images). Returns None
    if nothing suitable is found.
    """
    if repeats and trigger and (root / f"{repeats}_{trigger}").is_dir():
        return root / f"{repeats}_{trigger}"
    if (root / "train").is_dir():
        return root / "train"
    repeat_dirs = [d for d in root.iterdir() if d.is_dir() and re.match(r"^\d+_", d.name)]
    if len(repeat_dirs) == 1:
        return repeat_dirs[0]
    # Maybe root IS the train folder
    if any(f.suffix.lower() in SUPPORTED_EXTS for f in root.iterdir() if f.is_file()):
        return root
    return None


def run_validate_only(opts: dict) -> int:
    root = opts.get("output_dir") or opts.get("source_dir")
    if not root:
        print("\nERROR: --validate-only needs --output (dataset root or train folder).")
        return 1
    if not root.is_dir():
        print(f"\nERROR: Not a folder: {root}")
        return 1

    trigger = opts.get("trigger")
    if not trigger:
        # Try metadata.json next to or above the train folder
        for meta_dir in (root, root.parent):
            meta = meta_dir / "metadata.json"
            if meta.is_file():
                try:
                    trigger = json.loads(meta.read_text(encoding="utf-8")).get("trigger_token")
                    if trigger:
                        print(f"  Trigger token from metadata.json: {trigger}")
                        break
                except Exception:
                    pass
    if not trigger:
        print("\nERROR: --validate-only needs --trigger (no metadata.json with trigger_token found).")
        return 1

    train_dir = resolve_train_dir(root, trigger, opts.get("repeats"))
    if train_dir is None:
        print(f"\nERROR: No train folder (train/ or N_trigger/) with images found in: {root}")
        return 1

    print(f"\nValidating: {train_dir}  (trigger: {trigger}, no files will be changed)")
    run_quality = not opts.get("no_quality_check")
    warnings = validate_dataset(train_dir, trigger, run_quality=run_quality,
                                caption_mode=opts.get("captions"),
                                blur_threshold=opts.get("blur_threshold", BLUR_THRESHOLD))

    if warnings:
        print(f"\n  Validation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    WARN  {w}")
        return 1
    print("\n  Validation passed -- no issues found")
    return 0


# -- Main -----------------------------------------------------------------------

def main() -> int:
    opts = parse_args()

    if opts.get("validate_only"):
        return run_validate_only(opts)

    needs_interactive = not all([opts["source_dir"], opts["output_dir"], opts["lora_type"], opts["trigger"]])
    if needs_interactive:
        opts = run_interactive(opts)

    if opts.get("captions") is None:
        opts["captions"] = "template"
    if opts.get("move") is None:
        opts["move"] = False
    if opts.get("backup") is None:
        opts["backup"] = True

    source_dir: Path = opts["source_dir"]
    output_dir: Path = opts["output_dir"]

    # -- Validate source --
    if not source_dir.exists():
        print(f"\nERROR: Source folder does not exist: {source_dir}")
        return 1
    if not source_dir.is_dir():
        print(f"\nERROR: Source path is not a folder: {source_dir}")
        return 1

    # -- Check dependencies --
    if opts["captions"] == "vlm" and not has_vlm_support():
        print("\nERROR: VLM captioning requires the anthropic package.")
        print("  Install with: pip install anthropic")
        print("  Set ANTHROPIC_API_KEY environment variable.")
        return 1

    if opts["captions"] == "local" and not has_local_vlm_support():
        print("\nERROR: Local captioning requires transformers and torch.")
        print("  Install with: pip install transformers torch")
        return 1

    if opts.get("resize") and not has_pillow():
        print("\nERROR: --resize requires Pillow.")
        print("  Install with: pip install Pillow")
        return 1

    # -- Set derived paths --
    if opts.get("repeats"):
        train_folder_name = f"{opts['repeats']}_{opts['trigger']}"
    else:
        train_folder_name = "train"

    opts["output_dir"] = output_dir
    opts["train_dir"] = output_dir / train_folder_name
    opts["raw_dir"] = output_dir / "raw"

    # -- Validate trigger token --
    trigger = opts["trigger"]
    if not trigger:
        print("\nERROR: Trigger token cannot be empty.")
        return 1
    if not re.match(r'^[a-zA-Z0-9_-]+$', trigger):
        print(f"\nWARNING: Trigger token '{trigger}' contains special characters. "
              "Recommended: lowercase letters + numbers only (e.g. mychar01).")

    # -- Discover images --
    print(f"\nScanning: {source_dir}")
    found = discover_images(source_dir)

    if not found["all"]:
        print("ERROR: No image files found in source folder.")
        return 1

    # -- Handle WebP --
    images_to_process = list(found["supported"])
    if found["webp"]:
        if opts["convert_webp"]:
            if not has_pillow():
                print("WARNING: --convert-webp requires Pillow. Install with: pip install Pillow")
                print(f"  {len(found['webp'])} WebP files will be SKIPPED.")
            else:
                images_to_process += found["webp"]
                print(f"  {len(found['webp'])} WebP files will be converted to PNG")
        else:
            print(f"  {len(found['webp'])} WebP files found -- use --convert-webp to include them")

    if found["unsupported"]:
        print(f"  {len(found['unsupported'])} unsupported files will be ignored:")
        for f in found["unsupported"][:5]:
            print(f"    {f.name}")
        if len(found["unsupported"]) > 5:
            print(f"    ... and {len(found['unsupported']) - 5} more")

    images_to_process.sort()

    if not images_to_process:
        print("ERROR: No processable images found.")
        return 1

    print(f"  Found {len(images_to_process)} images to process")

    # -- Image quality pre-check --
    quality = None
    if not opts.get("no_quality_check"):
        print("  Running quality checks...")
        quality = check_images_quality(images_to_process,
                                       blur_threshold=opts.get("blur_threshold", BLUR_THRESHOLD))
        has_issues = (quality["too_small"] or quality["blurry"]
                      or quality["exact_duplicates"] or quality["near_duplicates"])
        if has_issues:
            if quality["too_small"]:
                print(f"  WARNING: {len(quality['too_small'])} images below {MIN_IMAGE_DIM}px")
            if quality["blurry"]:
                print(f"  WARNING: {len(quality['blurry'])} images may be blurry")
            if quality["exact_duplicates"]:
                total_dupes = sum(len(g) for g in quality["exact_duplicates"])
                print(f"  WARNING: {total_dupes} images in {len(quality['exact_duplicates'])} exact duplicate groups")
            if quality["near_duplicates"]:
                print(f"  WARNING: {len(quality['near_duplicates'])} near-duplicate pairs detected")
        else:
            print("  Quality checks passed")

    # -- Build plan --
    ops = build_plan(images_to_process, opts)

    # -- Always print dry-run --
    print_plan(ops, opts, quality)

    if not opts["execute"]:
        print("\n  (Dry-run complete. Nothing was changed.)")
        print("  Re-run with --execute to apply changes.\n")
        return 0

    # -- Warn if output directory already has files --
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"\n  WARNING: Output directory is not empty: {output_dir}")
        print("  Existing files may be mixed with new output.")
        if needs_interactive:
            confirm = ask("Continue anyway?", default="no", choices=["yes", "no"])
            if confirm != "yes":
                print("  Aborted.")
                return 1

    # -- Execute --
    print()
    print("=" * 60)
    print("  EXECUTING")
    print("=" * 60)
    stats = execute_plan(ops, opts)

    # -- Write README + metadata --
    run_quality = not opts.get("no_quality_check")
    warnings = validate_dataset(opts["train_dir"], trigger, run_quality=run_quality,
                                caption_mode=opts.get("captions"),
                                blur_threshold=opts.get("blur_threshold", BLUR_THRESHOLD))
    stats["validation_warnings"] = warnings
    write_readme(output_dir, opts, stats)
    write_metadata(output_dir, opts, stats)
    print(f"  OK  README.md")
    print(f"  OK  metadata.json")

    # -- Summary --
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_images = stats["copied"] + stats["moved"] + stats["converted"]
    print(f"  Images processed:  {total_images}")
    print(f"  Captions written:  {stats['captions_written']}")
    if stats["resized"]:
        print(f"  Images resized:    {stats['resized']}")
    if stats["converted"]:
        print(f"  WebP converted:    {stats['converted']}")
    if stats["vlm_captioned"]:
        print(f"  VLM captioned:     {stats['vlm_captioned']}")
    if stats["vlm_errors"]:
        print(f"  VLM errors:        {stats['vlm_errors']} (template fallback used)")
    if stats["errors"]:
        print(f"  Errors:            {len(stats['errors'])}")
        for e in stats["errors"]:
            print(f"    ERR  {e}")

    if warnings:
        print(f"\n  Validation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    WARN  {w}")
    else:
        print("\n  Validation passed -- no issues found")

    print()
    print(f"  Dataset ready at: {output_dir}")
    print(f"  Train folder:     {opts['train_dir']}")
    print(f"  Trigger token:    {trigger}")

    if opts["captions"] == "template":
        print()
        print("  NEXT STEP: Open train/*.txt and fill in the [BRACKET] placeholders.")
        print("  Replace [CLOTHING], [SHOT TYPE], [POSE/ACTION], [EXPRESSION], etc.")
        print("  with actual values for each image before starting training.")
        print("  See README.md for captioning strategy and format.")
    elif opts["captions"] in ("vlm", "local"):
        print()
        print("  NEXT STEP: Review auto-generated captions in train/*.txt.")
        if opts["lora_type"] == "character":
            print("  For character LoRA: remove permanent identity descriptions")
            print("  (hair color, eye color, skin tone) -- the trigger token learns these.")
        print("  Fix any inaccuracies. Run validation to confirm quality:")
        print("    python prepare_dataset.py --validate-only --output <dataset>")

    print()

    return 0 if not stats["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
