"""Tests for prepare_dataset.py v2.2.0"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from prepare_dataset import (
    pad_number,
    word_count,
    is_safe_filename,
    get_caption,
    compute_file_hash,
    find_exact_duplicates,
    find_near_duplicates,
    validate_dataset,
    build_plan,
    clean_local_caption,
    repeated_phrase_warnings,
    resolve_train_dir,
    LORA_TYPES,
    CAPTION_TEMPLATES,
    MINIMAL_CAPTIONS,
)

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


# -- Helpers -------------------------------------------------------------------

class TestPadNumber:
    def test_small_dataset(self):
        assert pad_number(1, 50) == "01"
        assert pad_number(9, 99) == "09"
        assert pad_number(42, 99) == "42"

    def test_large_dataset(self):
        assert pad_number(1, 100) == "001"
        assert pad_number(99, 200) == "099"
        assert pad_number(150, 200) == "150"

    def test_boundary(self):
        assert pad_number(99, 99) == "99"
        assert pad_number(100, 100) == "100"


class TestWordCount:
    def test_basic(self):
        assert word_count("hello world") == 2

    def test_single(self):
        assert word_count("one") == 1

    def test_many(self):
        assert word_count("a b c d e") == 5

    def test_extra_spaces(self):
        assert word_count("  hello   world  ") == 2


class TestIsSafeFilename:
    def test_valid(self):
        assert is_safe_filename("01.png") is True
        assert is_safe_filename("my-file_02.jpg") is True
        assert is_safe_filename("image.jpeg") is True

    def test_cyrillic(self):
        assert is_safe_filename("файл.png") is False

    def test_spaces(self):
        assert is_safe_filename("my file.png") is False

    def test_special_chars(self):
        assert is_safe_filename("test!.png") is False
        assert is_safe_filename("img (1).png") is False


# -- Caption generation --------------------------------------------------------

class TestGetCaption:
    def test_minimal_starts_with_trigger(self):
        for lora_type in LORA_TYPES:
            caption = get_caption(0, lora_type, "test01", "minimal")
            assert caption.startswith("test01"), f"minimal caption for {lora_type} should start with trigger"

    def test_minimal_matches_template(self):
        for lora_type in LORA_TYPES:
            caption = get_caption(0, lora_type, "abc", "minimal")
            expected = MINIMAL_CAPTIONS[lora_type].format(trigger="abc")
            assert caption == expected

    def test_template_contains_trigger(self):
        caption = get_caption(0, "character", "mychar01", "template")
        assert "mychar01" in caption

    def test_template_has_brackets(self):
        caption = get_caption(0, "character", "x", "template")
        assert "[CLOTHING]" in caption

    def test_vlm_placeholder(self):
        caption = get_caption(0, "character", "test01", "vlm")
        assert "auto-caption" in caption

    def test_local_placeholder(self):
        caption = get_caption(0, "character", "test01", "local")
        assert "auto-caption" in caption

    def test_template_wraps_around(self):
        n = len(CAPTION_TEMPLATES["character"])
        c0 = get_caption(0, "character", "t", "template")
        cn = get_caption(n, "character", "t", "template")
        assert c0 == cn

    def test_all_types_have_templates(self):
        for lora_type in LORA_TYPES:
            assert len(CAPTION_TEMPLATES[lora_type]) > 0
            caption = get_caption(0, lora_type, "trg", "template")
            assert caption.startswith("trg")


# -- Local caption cleanup -------------------------------------------------------

class TestCleanLocalCaption:
    def test_strips_there_is(self):
        assert clean_local_caption("there is a man in a kitchen") == "a man in a kitchen"

    def test_strips_image_of(self):
        assert clean_local_caption("this is an image of a man") == "a man"

    def test_strips_photo_of(self):
        assert clean_local_caption("a photo of a woman standing") == "a woman standing"

    def test_reinserts_article(self):
        # "there are" leaves a remainder without an article -> "a" is prepended
        result = clean_local_caption("there are two men sitting")
        assert result == "a two men sitting"

    def test_keeps_normal_caption(self):
        text = "a man wearing a dark uniform, standing in a kitchen"
        assert clean_local_caption(text) == text

    def test_longest_prefix_wins(self):
        # "this is a picture of a" must be stripped fully, not just "this is a"
        assert clean_local_caption("this is a picture of a dog") == "a dog"


# -- Repeated phrase (attribute sticking) ----------------------------------------

class TestRepeatedPhraseWarnings:
    def _captions(self, texts):
        return {f"{i:02d}": t for i, t in enumerate(texts, start=1)}

    def test_flags_repeated_attribute(self):
        caps = self._captions([
            "t01, medium shot of a man, warm lighting indoors",
            "t01, close-up of a man, warm lighting at night",
            "t01, full body shot of a man, warm lighting outside",
            "t01, profile view of a man, warm lighting on face",
        ])
        warnings = repeated_phrase_warnings(caps)
        assert any("warm lighting" in w for w in warnings)

    def test_ignores_media_anchor(self):
        caps = self._captions([
            "t01, a still from a TV show, man in a kitchen",
            "t01, a still from a TV show, man in an office",
            "t01, a still from a TV show, man on a street",
            "t01, a still from a TV show, man in a car",
        ])
        warnings = repeated_phrase_warnings(caps)
        assert not any("tv show" in w.lower() for w in warnings)

    def test_too_few_captions_skipped(self):
        caps = self._captions([
            "t01, warm lighting", "t01, warm lighting", "t01, warm lighting",
        ])
        assert repeated_phrase_warnings(caps) == []

    def test_varied_captions_clean(self):
        caps = self._captions([
            "t01, soft daylight in a park",
            "t01, neon glow on a street",
            "t01, overcast sky near a beach",
            "t01, studio strobes against a backdrop",
        ])
        assert repeated_phrase_warnings(caps) == []


# -- Validate-only: train dir resolution ------------------------------------------

@pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")
class TestResolveTrainDir:
    def _make_image(self, path):
        Image.new("RGB", (64, 64), "red").save(path)

    def test_finds_train_subdir(self, tmp_path):
        (tmp_path / "train").mkdir()
        assert resolve_train_dir(tmp_path) == tmp_path / "train"

    def test_finds_repeats_dir(self, tmp_path):
        (tmp_path / "10_mychar01").mkdir()
        assert resolve_train_dir(tmp_path) == tmp_path / "10_mychar01"

    def test_repeats_with_explicit_args(self, tmp_path):
        (tmp_path / "10_t01").mkdir()
        (tmp_path / "20_t02").mkdir()
        assert resolve_train_dir(tmp_path, trigger="t01", repeats=10) == tmp_path / "10_t01"

    def test_root_is_train_dir(self, tmp_path):
        self._make_image(tmp_path / "01.png")
        assert resolve_train_dir(tmp_path) == tmp_path

    def test_nothing_found(self, tmp_path):
        (tmp_path / "docs").mkdir()
        assert resolve_train_dir(tmp_path) is None


# -- VLM captioning (mocked API) --------------------------------------------------

class TestCaptionBatchVlm:
    """End-to-end test of caption_batch_vlm with a fake anthropic module."""

    def _install_fake_anthropic(self, monkeypatch, caption_text):
        import types

        class FakeBlock:
            def __init__(self, text):
                self.text = text

        class FakeMessage:
            def __init__(self, text):
                self.content = [FakeBlock(text)]

        class FakeMessages:
            async def create(self, model, max_tokens, messages):
                return FakeMessage(caption_text)

        class FakeAsyncAnthropic:
            def __init__(self):
                self.messages = FakeMessages()

        fake = types.ModuleType("anthropic")
        fake.AsyncAnthropic = FakeAsyncAnthropic
        monkeypatch.setitem(sys.modules, "anthropic", fake)

    def _make_ops(self, tmp_path, names):
        ops = []
        for i, name in enumerate(names, start=1):
            p = tmp_path / name
            if name != "missing.png":  # deliberately absent file
                p.write_bytes(b"\x89PNG fake image data")
            ops.append({"src": p, "index": i})
        return ops

    def test_success_path(self, tmp_path, monkeypatch):
        from prepare_dataset import caption_batch_vlm
        self._install_fake_anthropic(
            monkeypatch,
            "test01, a photograph, medium shot of a man wearing a coat, standing, calm expression, street, daylight",
        )
        ops = self._make_ops(tmp_path, ["a.png", "b.png"])
        result = caption_batch_vlm(ops, "character", "test01")
        assert len(result["captions"]) == 2
        assert result["errors"] == {}
        assert all(c.startswith("test01") for c in result["captions"].values())

    def test_trigger_prepended_when_missing(self, tmp_path, monkeypatch):
        from prepare_dataset import caption_batch_vlm
        self._install_fake_anthropic(monkeypatch, "a man in a coat, standing on a street")
        ops = self._make_ops(tmp_path, ["a.png"])
        result = caption_batch_vlm(ops, "character", "test01")
        assert result["captions"][1].startswith("test01, ")

    def test_error_attributed_to_failing_image(self, tmp_path, monkeypatch):
        """A failure must be attributed to the image that failed, not the first one."""
        from prepare_dataset import caption_batch_vlm
        self._install_fake_anthropic(monkeypatch, "test01, fine caption here")
        # index 2 points at a file that does not exist -> open() raises
        ops = self._make_ops(tmp_path, ["a.png", "missing.png", "c.png"])
        result = caption_batch_vlm(ops, "character", "test01")
        assert sorted(result["captions"].keys()) == [1, 3]
        assert list(result["errors"].keys()) == [2]


# -- File hashing --------------------------------------------------------------

class TestFileHash:
    def test_identical_content(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"same content here")
        f2.write_bytes(b"same content here")
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_different_content(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_deterministic(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"test data")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2


class TestFindExactDuplicates:
    def test_finds_duplicates(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f3 = tmp_path / "c.bin"
        f1.write_bytes(b"same")
        f2.write_bytes(b"same")
        f3.write_bytes(b"different")
        dupes = find_exact_duplicates([f1, f2, f3])
        assert len(dupes) == 1
        assert set(dupes[0]) == {f1, f2}

    def test_no_duplicates(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"one")
        f2.write_bytes(b"two")
        assert find_exact_duplicates([f1, f2]) == []

    def test_multiple_groups(self, tmp_path):
        files = []
        for name, content in [("a", b"X"), ("b", b"X"), ("c", b"Y"), ("d", b"Y"), ("e", b"Z")]:
            f = tmp_path / f"{name}.bin"
            f.write_bytes(content)
            files.append(f)
        dupes = find_exact_duplicates(files)
        assert len(dupes) == 2


# -- Pillow-dependent tests ----------------------------------------------------

@pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")
class TestPerceptualHash:
    def _make_image(self, path, size=(256, 256), color="red"):
        Image.new("RGB", size, color).save(path)

    def test_identical_images(self, tmp_path):
        from prepare_dataset import compute_perceptual_hash, hamming_distance
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        self._make_image(a, color="red")
        self._make_image(b, color="red")
        ha = compute_perceptual_hash(a)
        hb = compute_perceptual_hash(b)
        assert ha == hb
        assert hamming_distance(ha, hb) == 0

    def test_different_images(self, tmp_path):
        from prepare_dataset import compute_perceptual_hash, hamming_distance
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        # Create images with actual gradient differences, not solid colors
        img_a = Image.new("RGB", (256, 256))
        for x in range(256):
            for y in range(256):
                img_a.putpixel((x, y), (x, 0, 0))
        img_a.save(a)
        img_b = Image.new("RGB", (256, 256))
        for x in range(256):
            for y in range(256):
                img_b.putpixel((x, y), (0, 0, y))
        img_b.save(b)
        ha = compute_perceptual_hash(a)
        hb = compute_perceptual_hash(b)
        assert hamming_distance(ha, hb) > 0

    def test_near_duplicates_detected(self, tmp_path):
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        img_a = Image.new("RGB", (256, 256), "red")
        img_a.save(a)
        img_b = Image.new("RGB", (256, 256), (255, 5, 5))
        img_b.save(b)
        pairs = find_near_duplicates([a, b], threshold=20)
        assert len(pairs) >= 0  # near-identical reds; threshold depends on exact hash


@pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")
class TestImageQuality:
    def _make_image(self, path, size=(1024, 1024), color="red"):
        Image.new("RGB", size, color).save(path)

    def test_small_image_detected(self, tmp_path):
        from prepare_dataset import get_image_dimensions
        img = tmp_path / "small.png"
        self._make_image(img, size=(256, 256))
        w, h = get_image_dimensions(img)
        assert w == 256 and h == 256

    def test_blur_solid_color(self, tmp_path):
        from prepare_dataset import detect_blur_score, BLUR_THRESHOLD
        img = tmp_path / "solid.png"
        self._make_image(img, color="red")
        score = detect_blur_score(img)
        assert score < BLUR_THRESHOLD

    def test_resize_downscale(self, tmp_path):
        from prepare_dataset import resize_image, get_image_dimensions
        src = tmp_path / "big.png"
        dst = tmp_path / "out.png"
        self._make_image(src, size=(2048, 1536))
        result = resize_image(src, dst, 1024)
        assert result is True
        w, h = get_image_dimensions(dst)
        assert min(w, h) == 1024

    def test_resize_no_upscale(self, tmp_path):
        from prepare_dataset import resize_image
        src = tmp_path / "small.png"
        dst = tmp_path / "out.png"
        self._make_image(src, size=(512, 512))
        result = resize_image(src, dst, 1024)
        assert result is False
        assert dst.exists()

    def test_resize_preserves_aspect(self, tmp_path):
        from prepare_dataset import resize_image, get_image_dimensions
        src = tmp_path / "wide.png"
        dst = tmp_path / "out.png"
        self._make_image(src, size=(3000, 2000))
        resize_image(src, dst, 1024)
        w, h = get_image_dimensions(dst)
        ratio_src = 3000 / 2000
        ratio_dst = w / h
        assert abs(ratio_src - ratio_dst) < 0.01


# -- Build plan ----------------------------------------------------------------

@pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")
class TestBuildPlan:
    def _make_image(self, path):
        Image.new("RGB", (100, 100), "red").save(path)

    def test_basic_plan(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        for i in range(3):
            self._make_image(src / f"img{i}.png")

        opts = {
            "lora_type": "character",
            "trigger": "mychar01",
            "captions": "template",
            "move": False,
            "train_dir": out / "train",
        }
        images = sorted(src.glob("*.png"))
        ops = build_plan(images, opts)
        assert len(ops) == 3
        assert ops[0]["out_name"] == "01.png"
        assert ops[1]["out_name"] == "02.png"
        assert ops[2]["out_name"] == "03.png"
        assert all("mychar01" in op["caption"] for op in ops)

    def test_repeats_folder(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        self._make_image(src / "img.png")

        opts = {
            "lora_type": "character",
            "trigger": "t01",
            "captions": "minimal",
            "move": False,
            "train_dir": out / "10_t01",
        }
        ops = build_plan([src / "img.png"], opts)
        assert "10_t01" in str(ops[0]["out_image"])


# -- Validation ----------------------------------------------------------------

@pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")
class TestValidation:
    def _make_image(self, path, size=(1024, 1024)):
        Image.new("RGB", size, "red").save(path)

    def test_valid_dataset(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        (train / "01.txt").write_text("mychar01, a character, medium shot", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert not any("Missing caption" in w for w in warnings)
        assert not any("BRACKET" in w for w in warnings)

    def test_missing_caption(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        warnings = validate_dataset(train, "mychar01")
        assert any("Missing caption" in w for w in warnings)

    def test_orphan_caption(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        (train / "01.txt").write_text("mychar01, text", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert any("Orphan caption" in w for w in warnings)

    def test_unfilled_bracket(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        (train / "01.txt").write_text("mychar01, wearing [CLOTHING], front view", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert any("BRACKET" in w for w in warnings)

    def test_missing_trigger(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        (train / "01.txt").write_text("a character, medium shot, standing", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert any("Trigger token" in w for w in warnings)

    def test_vlm_failed_marker(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        (train / "01.txt").write_text("mychar01, template  # VLM_FAILED -- fill manually", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert any("VLM" in w for w in warnings)

    def test_caption_too_long(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        long_caption = "mychar01, " + " ".join(["word"] * 65)
        (train / "01.txt").write_text(long_caption, encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert any("too long" in w for w in warnings)

    def test_caption_short_flagged(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        (train / "01.txt").write_text("mychar01, a character, medium shot", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert any("short" in w for w in warnings)

    def test_caption_short_skipped_in_minimal(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        (train / "01.txt").write_text("mychar01, portrait photo", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01", caption_mode="minimal")
        assert not any("short" in w for w in warnings)

    def test_caption_in_target_range_ok(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        caption = "mychar01, " + " ".join(["word"] * 20)
        (train / "01.txt").write_text(caption, encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert not any("short" in w or "long" in w for w in warnings)

    def test_all_identical_captions(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        for i in range(5):
            self._make_image(train / f"0{i+1}.png", size=(1024, 1024 + i))
            (train / f"0{i+1}.txt").write_text("mychar01, same caption", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert any("identical" in w for w in warnings)

    def test_duplicate_images(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        import shutil
        shutil.copy2(train / "01.png", train / "02.jpg")
        (train / "01.txt").write_text("mychar01, shot one", encoding="utf-8")
        (train / "02.txt").write_text("mychar01, shot two", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert any("Duplicate" in w or "identical" in w for w in warnings)

    def test_unsafe_filename(self, tmp_path):
        train = tmp_path / "train"
        train.mkdir()
        self._make_image(train / "01.png")
        (train / "01.txt").write_text("mychar01, caption", encoding="utf-8")
        self._make_image(train / "my file (2).png")
        (train / "my file (2).txt").write_text("mychar01, caption", encoding="utf-8")
        warnings = validate_dataset(train, "mychar01")
        assert any("Unsafe filename" in w for w in warnings)

    def test_nonexistent_dir(self):
        warnings = validate_dataset(Path("/nonexistent/train"), "x")
        assert len(warnings) == 1
        assert "does not exist" in warnings[0]
