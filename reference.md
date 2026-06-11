# LoRA Captioning & Dataset Reference

Full reference for `lora-dataset-prep`. Read this when writing or reviewing captions,
choosing dataset images, or debugging a LoRA that "sticks" attributes.

---

## Character LoRA — Full Guide

Primary target for this skill. Based on Ostris AI Toolkit, the Z-Image official guide,
Civitai dataset/Character LoRA guides, and community Z-Image Turbo references.

### 1. Images

| Parameter | Rule |
|---|---|
| **Count** | **15–30** images (optimum ~20). Below 15 → overfit risk; above 50 → fine-tuning territory |
| **Resolution** | **1024×1024** native for Z-Image Turbo. Minimum **768px** on the short side |
| **Format** | PNG or high-quality JPEG. No watermarks, text overlays, or heavy filters |
| **Quality** | Sharp, well-lit. No blur, compression artifacts, or blown highlights |

**Required shot variety** — cover as many as possible:

- Front portrait
- 3/4 view (both directions)
- Profile
- Face close-up
- Medium shot (waist-up)
- Full body
- Different facial expressions
- Different lighting
- Different backgrounds

**Forbidden:**

- Duplicates or near-duplicates (minimal angle change between frames)
- Cropped or obscured faces
- Extreme angles / lens distortion
- Multiple subjects (when training one character)
- Low resolution

Use `--resize 1024` to normalize size. The script flags blur and near-duplicates automatically.

### 2. Captioning principle

> **Describe what you do NOT want the LoRA to learn.**

Everything **not** in the caption gets bound to the trigger token. Therefore:

- **Do NOT** describe face shape, eye color, skin tone, body type — identity is learned via trigger
- **DO** describe clothing, pose, shot type, expression, camera angle, background, lighting, media type — so these stay flexible at inference

**Example logic:** If every photo shows a police uniform and you never write `wearing a police uniform`, the model treats the uniform as part of identity and always generates it.

### 3. Caption structure

**Format:** natural language sentences — **not** comma-separated SD 1.5 tag soup. Z-Image Turbo is trained on natural-language prompts.

**Formula (same order in every caption):**

```
[trigger], [media type], [shot type] of a [man/woman], [clothing], [pose/action],
[expression], [background/setting], [lighting]
```

| Constraint | Value |
|---|---|
| Length | **15–35 words** (1–2 sentences). Longer → overfit risk |
| Language | Natural English, simple and descriptive |
| Consistency | Same element order in every `.txt` file |

### 4. What to include

| Element | Why | Example |
|---|---|---|
| Trigger token | Identity anchor | `zvqmark` — first word, once |
| Media / style type | Prevents binding TV/film aesthetic to character | `a still from a TV show`, `a photograph` |
| Shot type | Prevents one framing from sticking | `close-up`, `medium shot`, `full body shot` |
| Clothing | Keeps outfit flexible | `wearing a dark police uniform with a blue shirt` |
| Pose / action | Separates character from pose | `standing with hands on hips`, `sitting and eating` |
| Expression | Flexible at inference | `serious expression`, `smiling`, `surprised` |
| Camera angle | Prevents one angle from sticking | `front view`, `profile view`, `seen from behind` |
| Background | Separates location from identity | `in a kitchen`, `office with a pennant` |
| Lighting | Separates lighting from identity | `warm indoor lighting`, `bright studio lighting` |

### 5. What to omit

| Do not write | Why |
|---|---|
| Face features (nose, eyes, cheekbones) | Learned via trigger |
| Eye color | Identity |
| Skin tone / type | Identity |
| Hair color / style | Omit if fixed identity; include only if you want it changeable at inference |
| Body type / height | Identity (unless you want it flexible) |
| Quality tags (`masterpiece`, `best quality`, `4k`) | Useless for training, clutters caption |
| Same adjective in every caption | Causes sticking (e.g. `warm` in 30/47 captions → always warm) |
| Poetic / artistic language | Keep captions plain and factual |

### 6. Good vs bad example

**Bad:**

```
zvqmark, a man wearing a dark police uniform with a blue shirt and tie, standing indoors in a kitchen, hands on hips, medium shot, looking down, warm interior lighting
```

**Good:**

```
zvqmark, a still from a TV show, medium shot of a man wearing a dark police uniform with a blue shirt and tie, standing in a kitchen with hands on hips, looking down with a stern expression, warm interior lighting
```

What changed: added **media type**, added **expression**, restructured as natural language with consistent element order.

### 7. Typical mistakes

| Mistake | Effect |
|---|---|
| Near-duplicate frames | Overfit on specific pose/angle combos |
| Other people in frame, unlabeled | Model can't tell who is the trigger character |
| Missing media type on TV/film stills | TV aesthetic becomes part of identity |
| Same lighting adjective everywhere | Lighting sticks across all generations |
| Captions over 35 words | Higher overfit risk |
| Filler phrases (`there is a`, `a photo of`) from BLIP `local` mode | Wastes the word budget, no useful signal — rewrite per formula |
| Trusting raw BLIP `local` captions as final | Generic, formula-violating; always rewrite or use `vlm`/Florence-2 |
| No captions at all | Worse than minimal captions; worse than over-captioning |

### 8. Quick reference

```
FORMULA:
[trigger], [media type], [shot type] of a man/woman, [clothing], [pose/action],
[expression], [background/setting], [lighting]

INCLUDE:  clothing, pose, expression, angle, background, lighting, media type
OMIT:     face, eyes, skin, quality tags, repeated adjectives
LENGTH:   15–35 words
LANGUAGE: natural language, not tags
TRIGGER:  first word, once
```

Template mode (`--captions template`) generates captions following this formula with `[BRACKET]` placeholders — fill them per image before training. VLM mode (`--captions vlm`) applies these rules automatically.

---

## Caption Strategy — Other LoRA Types

### Style LoRA

**Rule:** Caption the CONTENT, not the style. Let the LoRA learn "how" from what varies.

```
stlmono, a landscape with mountains, soft light
stlmono, a portrait of a person, neutral background
stlmono, a still life with fruits, warm tones
```

Dataset should contain **diverse subjects** all rendered in the target style.
Adding style hints (`flat colors`, `soft lines`) is OK if minimal.
Never over-describe the style in captions — the LoRA learns it from the images.

### Object / Product LoRA

**Rule:** Trigger token + object class (semantic anchor).

```
zbxshoe, a white running shoe with orange sole, front view
zbxshoe, a white running shoe with orange sole, side view
zbxshoe, a white running shoe with orange sole, sole detail
```

Cover: many angles (front/side/back/top), lighting scenarios, close-ups of distinctive details,
in-use context shots.

### Clothing LoRA

**Rule:** Describe the garment specifically; wearer is generic.

```
zbxjacket, a black leather jacket, worn by a woman, front view, full body
zbxjacket, a black leather jacket, flat lay, overhead, white background
zbxjacket, a black leather jacket, detail of zipper, close-up
```

Show: on-body from multiple angles, flat lay, detail shots, different wearers if possible.

### Environment / Scene LoRA

**Rule:** Describe the varying elements; location is the constant.

```
envtown, street scene, morning light, cobblestone detail
envtown, wide establishing shot, overcast sky, city background
envtown, interior corridor, warm artificial lighting, evening
```

Include: wide establishing + detail shots, different times of day/lighting, same location.

---

## Local Captioning (BLIP / Florence-2) — Read Before Using

`--captions local` runs a vision model on your machine. **The default model
(`Salesforce/blip-image-captioning-large`) is a plain caption model — it does NOT
follow the caption formula.** It emits short, generic strings such as
`there is a man in a kitchen`, `this is an image of a man`, or `two men sitting on
a couch`. These are missing media type, shot type, expression, lighting, and
clothing detail, and may name multiple subjects.

Consequences for character LoRA:

- Missing **media type** → the TV/film look gets baked into the character.
- Missing **shot type / expression / lighting** → those attributes stick at inference.
- **Multiple subjects** named (`a man and a woman`) → the model can't isolate the trigger.

The script strips obvious filler prefixes (`there is a`, `this is an image of`)
and validation flags remaining filler and multi-subject captions. Even so:

> **For character LoRA, BLIP `local` output must be rewritten by hand (or regenerated
> with `--captions vlm`, or a Florence-2 local model via
> `--local-model microsoft/Florence-2-large`).** Treat raw BLIP captions as a rough
> first draft only, never as final training captions.

---

## Validation Checks — Full List

The script validates after execution (or standalone via `--validate-only`) and reports warnings:

- Every image has a matching `.txt` file
- Every `.txt` has a matching image
- No duplicate base names
- Trigger token appears in every caption
- Every caption starts with the trigger token
- Empty captions reported
- Captions < 15 words flagged (short; skipped in `minimal` mode)
- Captions > 35 words flagged (long)
- Captions > 60 words flagged (too long for training)
- Identical/low-diversity captions warned
- Unsafe filenames flagged (spaces, non-ASCII, special chars)
- Unfilled `[BRACKET]` placeholders flagged
- Filler phrases flagged (`there is a`, `this is an image of`, `a photo of` — typical of plain BLIP output)
- Multiple-subject captions flagged (`two men`, `a man and a woman`, `another person` — fatal for single-subject LoRA)
- Attribute-sticking flagged: any content phrase (e.g. `look down`, `warm lighting`) repeated in ≥50% of captions (structural/shot-type words and the media-type anchor are ignored)
- VLM_FAILED markers flagged
- Image dimensions checked (warn if < 512px)
- Blur detection on output images (tune with `--blur-threshold`, 0 disables)
- Exact duplicate images detected by SHA-256 hash
- Near-duplicate images detected by perceptual hash (dHash, hamming distance)

### Blur detection note

The blur score is a heuristic (edge variance on a 256px thumbnail), not a calibrated
optical measurement. Smooth, well-lit studio portraits or images with large flat
backgrounds can score below the default threshold of 100 while being perfectly sharp.
Treat `BLUR` warnings as "look at this image", not "delete this image". If your
dataset style triggers many false positives, lower the threshold
(`--blur-threshold 50`) or disable it (`--blur-threshold 0`).
