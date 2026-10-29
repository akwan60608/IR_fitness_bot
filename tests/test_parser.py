import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import RapidOcrOptions

from chunking.pipeline.parser import PICTURE_PROMPTS, build_converter, parse_pdf


class ParserTest(unittest.TestCase):
    def test_describe_mode_uses_low_memory_picture_options(self) -> None:
        converter = build_converter(device="cpu", picture_mode="describe")
        options = converter.format_to_options[InputFormat.PDF].pipeline_options

        self.assertTrue(options.do_picture_classification)
        self.assertTrue(options.do_picture_description)
        self.assertEqual(options.picture_description_options.batch_size, 1)
        self.assertEqual(options.picture_description_options.scale, 2.0)
        self.assertEqual(options.picture_description_options.picture_area_threshold, 0.01)
        self.assertEqual(options.picture_description_options.prompt, PICTURE_PROMPTS["p0"])

    def test_describe_mode_accepts_the_numeric_evidence_prompt(self) -> None:
        converter = build_converter(device="cpu", picture_mode="describe", picture_prompt_profile="p1")
        options = converter.format_to_options[InputFormat.PDF].pipeline_options

        self.assertEqual(options.picture_description_options.prompt, PICTURE_PROMPTS["p1"])

    def test_p2_defers_visual_description_until_context_is_available(self) -> None:
        converter = build_converter(device="cpu", picture_mode="describe", picture_prompt_profile="p2")
        options = converter.format_to_options[InputFormat.PDF].pipeline_options

        self.assertTrue(options.do_picture_classification)
        self.assertFalse(options.do_picture_description)

    def test_passes_accelerator_options_to_converter(self) -> None:
        document = object()
        with patch("chunking.pipeline.parser.build_converter") as build_converter:
            build_converter.return_value.convert.return_value.document = document
            with patch.object(Path, "is_file", return_value=True):
                result = parse_pdf("example.pdf", device="cpu", num_threads=2)

        self.assertIs(result, document)
        build_converter.assert_called_once_with(
            "cpu",
            2,
            picture_mode="classify",
            picture_prompt_profile="p0",
            ocr_mode=ANY,
        )

    def test_uses_onnx_runtime_for_ocr(self) -> None:
        converter = build_converter(device="cuda")
        options = converter.format_to_options[InputFormat.PDF].pipeline_options

        self.assertIsInstance(options.ocr_options, RapidOcrOptions)
        self.assertEqual(options.ocr_options.backend, "onnxruntime")


if __name__ == "__main__":
    unittest.main()
