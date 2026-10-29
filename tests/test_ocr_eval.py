import unittest

from chunking.evals.ocr import evaluate_artifacts, normalize_text, validate_ground_truth


SOURCE_SHA = "source-sha"


def _sample(**overrides):
    raw = overrides.pop("ground_truth_raw", "Ａ　１")
    sample = {
        "schema_version": 1,
        "sample_id": "sample-1",
        "source_sha256": SOURCE_SHA,
        "page_no": 1,
        "coordinate_origin": "TOPLEFT",
        "bbox_normalized": {"l": 0, "t": 0, "r": 1000, "b": 1000},
        "target": "ocr",
        "region_type": "paragraph",
        "ground_truth_raw": raw,
        "ground_truth_normalized": normalize_text(raw),
        "critical_tokens": ["1"],
        "annotation_status": "source_visual_checked",
        "note": "test",
    }
    sample.update(overrides)
    return sample


def _location(page_no=1):
    return {
        "page_no": page_no,
        "bbox": {"l": 0, "t": 100, "r": 100, "b": 0, "coord_origin": "BOTTOMLEFT"},
    }


def _manifest(status="success"):
    return {
        "pipeline_run_id": "run-test",
        "source_sha256": SOURCE_SHA,
        "status": status,
        "config": {"parser": {"ocr": {"engine": "rapidocr"}}},
    }


class OcrEvaluationTest(unittest.TestCase):
    def test_normalize_keeps_punctuation_but_normalizes_unicode_and_whitespace(self) -> None:
        self.assertEqual(normalize_text("Ａ　１\n甲，乙。"), "A 1 甲,乙。")

    def test_rejects_mismatched_source_and_invalid_bbox(self) -> None:
        wrong_source = _sample(source_sha256="other")
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            validate_ground_truth([wrong_source], SOURCE_SHA)

        invalid_bbox = _sample(bbox_normalized={"l": 0, "t": 0, "r": 1001, "b": 10})
        with self.assertRaisesRegex(ValueError, "0..1000"):
            validate_ground_truth([invalid_bbox], SOURCE_SHA)

        missing_required = _sample()
        missing_required.pop("note")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_ground_truth([missing_required], SOURCE_SHA)

    def test_ocr_only_uses_same_page_records_and_scores_exact_match(self) -> None:
        report = evaluate_artifacts(
            manifest=_manifest(),
            samples=[_sample()],
            parsed_records=[
                {"doc_item_ref": "#/texts/1", "text": "A 1", "source_locations": [_location(1)]},
                {"doc_item_ref": "#/texts/2", "text": "wrong page", "source_locations": [_location(2)]},
            ],
            media_records=[],
            media_chunks=[],
            page_dimensions={1: (100, 100), 2: (100, 100)},
        )

        result = report["samples"][0]
        self.assertEqual(result["selected_doc_item_refs"], ["#/texts/1"])
        self.assertTrue(result["exact_match"])
        self.assertEqual(report["metrics"]["ocr"]["micro_cer"], 0.0)
        self.assertEqual(report["metrics"]["ocr"]["critical_token_accuracy"], 1.0)

    def test_missing_region_is_a_detection_miss(self) -> None:
        report = evaluate_artifacts(
            manifest=_manifest(),
            samples=[_sample()],
            parsed_records=[],
            media_records=[],
            media_chunks=[],
            page_dimensions={1: (100, 100)},
        )

        result = report["samples"][0]
        self.assertFalse(result["detected"])
        self.assertIn("not_detected", result["failure_reasons"])
        self.assertFalse(result["critical_tokens"]["1"])
        self.assertEqual(report["metrics"]["ocr"]["detection_recall"], 0.0)

    def test_visual_reading_order_does_not_change_ocr_metrics(self) -> None:
        visual = _sample(
            sample_id="visual-flow",
            target="visual",
            region_type="flowchart",
            ground_truth_raw="甲乙",
            ground_truth_normalized="甲乙",
            critical_tokens=["甲"],
            expected_facts=["甲", "乙"],
            expected_sequence=["甲", "乙"],
            expected_p2_kind="flow_chart",
        )
        report = evaluate_artifacts(
            manifest=_manifest(),
            samples=[_sample(), visual],
            parsed_records=[{"doc_item_ref": "#/texts/1", "text": "A 1", "source_locations": [_location()]}],
            media_records=[{
                "media_id": "media-1",
                "picture_ref": "#/pictures/1",
                "classification": {"label": "flow_chart"},
                "p2": {"kind": "flow_chart", "status": "generated"},
                "description": {"text": "乙甲"},
                "source_locations": [_location()],
            }],
            media_chunks=[{"media_id": "media-1", "text": "乙甲"}],
            page_dimensions={1: (100, 100)},
        )

        visual_result = next(result for result in report["samples"] if result["sample_id"] == "visual-flow")
        self.assertFalse(visual_result["reading_order_match"])
        self.assertIn("reading_order_mismatch", visual_result["failure_reasons"])
        self.assertEqual(report["metrics"]["ocr"]["micro_cer"], 0.0)
        self.assertEqual(report["metrics"]["visual"]["reading_order_pass_rate"], 0.0)

    def test_separates_description_coverage_from_contextual_representation(self) -> None:
        visual = _sample(
            sample_id="visual-context",
            target="visual",
            region_type="exercise_figure",
            expected_facts=["瑞士球", "伏地挺身"],
            expected_p2_kind="exercise_instruction",
        )
        report = evaluate_artifacts(
            manifest=_manifest(),
            samples=[visual],
            parsed_records=[],
            media_records=[{
                "media_id": "media-1",
                "picture_ref": "#/pictures/1",
                "classification": {"label": "photograph"},
                "p2": {"kind": "exercise_instruction", "status": "generated"},
                "description": {"text": "人物在瑞士球上。"},
                "source_locations": [_location()],
            }],
            media_chunks=[{"media_id": "media-1", "text": "人物在瑞士球上。\nDocument context: 伏地挺身"}],
            page_dimensions={1: (100, 100)},
        )

        result = report["samples"][0]
        self.assertEqual(result["description_fact_coverage"], 0.5)
        self.assertEqual(result["contextual_representation_fact_coverage"], 1.0)
        self.assertEqual(result["context_lift"], 0.5)

    def test_tables_are_scored_outside_media_metrics(self) -> None:
        table = _sample(
            sample_id="table-1",
            target="table",
            region_type="table",
            expected_facts=["优质蛋白", "牛肉"],
        )
        report = evaluate_artifacts(
            manifest=_manifest(),
            samples=[table],
            parsed_records=[],
            media_records=[],
            media_chunks=[],
            table_records=[{
                "table_id": "table-1",
                "table_ref": "#/tables/1",
                "markdown": "| 优质蛋白 |\\n| 牛肉 |",
                "source_locations": [_location()],
            }],
            page_dimensions={1: (100, 100)},
        )

        self.assertEqual(report["metrics"]["visual"]["sample_count"], 0)
        self.assertEqual(report["metrics"]["table"]["table_detection_recall"], 1.0)
        self.assertEqual(report["metrics"]["table"]["fact_coverage"], 1.0)

    def test_failed_manifest_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "successful pipeline manifest"):
            evaluate_artifacts(
                manifest=_manifest(status="failed"),
                samples=[_sample()],
                parsed_records=[],
                media_records=[],
                media_chunks=[],
                page_dimensions={1: (100, 100)},
            )


if __name__ == "__main__":
    unittest.main()
